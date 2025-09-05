# -*- coding: utf-8 -*-
"""
MoDL（Model-based Deep Learning）重构模块 - 适配新JSON配置结构
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import json
from typing import Dict, Tuple, Optional


class CombinedLoss(nn.Module):
    """组合损失函数（MSE）"""

    def __init__(self,l1_weight=0.5,l2_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, pred, target):
        return self.l2_weight * self.mse(pred, target) + self.l1_weight * self.l1_loss(pred,target)


class DataConsistency(nn.Module):
    """数据一致性模块"""

    def __init__(self, lambda_init=0.1):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, u, y, mask):
        kspace_pred = fft.fft2(u.squeeze(1))
        residual = torch.where(mask, kspace_pred - y, 0)
        grad = fft.ifft2(residual).real.unsqueeze(1)
        return u - self.lambda_ * grad


class RegularizationModule(nn.Module):
    """正则化模块"""

    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)


class IterBlock(nn.Module):
    """迭代块（数据一致性+正则化）"""

    def __init__(self, channels=64):
        super().__init__()
        self.dc = DataConsistency()
        self.reg = RegularizationModule(channels)

    def forward(self, u, y, mask):
        u_dc = self.dc(u, y, mask)
        u_reg = self.reg(u_dc)
        return u_reg


class MoDLRecon(nn.Module):
    """MoDL 主模型"""

    def __init__(self, num_iters=8, channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([IterBlock(channels) for _ in range(num_iters)])

    def forward(self, y, mask):
        u = fft.ifft2(y).real.unsqueeze(1)
        for block in self.blocks:
            u = block(u, y, mask)
        return u.squeeze(1)


class CustomCSMRIDataset(Dataset):
    """自定义 MRI 数据集 - 适配新数据路径结构"""

    def __init__(self, config: Dict, mode: str = 'train'):
        self.image_size = config["MoDL"]["data"]["image_size"]

        if mode == 'train':
            self.kspace_dir = config["MoDL"]["DATA_PATHS"]["TRAIN_NOISY"]
            self.mask_dir = config["MoDL"]["DATA_PATHS"]["TRAIN_MASK"]
            self.full_dir = config["MoDL"]["DATA_PATHS"]["TRAIN_CLEAN"]
        else:  # val
            self.kspace_dir = config["MoDL"]["DATA_PATHS"]["VAL_NOISY"]
            self.mask_dir = config["MoDL"]["DATA_PATHS"]["VAL_MASK"]
            self.full_dir = config["MoDL"]["DATA_PATHS"]["VAL_CLEAN"]

        self.file_list = [f for f in os.listdir(self.full_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        base_name = os.path.splitext(img_name)[0]

        # 加载图像
        img = Image.open(os.path.join(self.full_dir, img_name)).convert('L')
        img = transforms.Resize((self.image_size, self.image_size))(img)
        x = torch.from_numpy(np.array(img)).float() / 255.0

        # 加载 k-space
        kspace = np.load(os.path.join(self.kspace_dir, f"{base_name}_kspace.npy"))
        kspace = np.squeeze(kspace)
        y = torch.from_numpy(kspace).type(torch.complex64)

        # 加载掩码
        mask = np.load(os.path.join(self.mask_dir, f"{base_name}_mask.npy"))
        mask = np.squeeze(mask)
        mask = torch.tensor(mask, dtype=torch.bool)

        assert y.shape == (self.image_size, self.image_size)
        assert mask.shape == (self.image_size, self.image_size)
        assert x.shape == (self.image_size, self.image_size)

        return y, mask, x  # y: (H,W), mask: (H,W), x: (H,W)


class MoDLTrainer:
    """MoDL 训练器 - 适配新配置结构"""

    def __init__(self, config_path: str):
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 设备设置
        self.device = torch.device(self.config["MoDL"].get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # 初始化模型
        modl_config = self.config["MoDL"]
        self.model = MoDLRecon(
            num_iters=modl_config["optim"]["num_iters"],
            channels=modl_config["optim"]["channels"]
        ).to(self.device)

        # 损失函数
        self.criterion = CombinedLoss()

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=modl_config["train"]["learning_rate"],
            weight_decay=modl_config.get("WEIGHT_DECAY", 0.0)
        )

        # 学习率调度器
        self.scheduler = None

        # 创建输出目录
        os.makedirs(modl_config.get("OUTPUT_DIR", "./output"), exist_ok=True)

    def load_datasets(self):
        """加载数据集 - 适配新数据路径结构"""
        train_dataset = CustomCSMRIDataset(self.config, mode='train')
        val_dataset = CustomCSMRIDataset(self.config, mode='val')

        modl_config = self.config["MoDL"]
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=modl_config["train"]["batch_size"],
            shuffle=True,
            num_workers=modl_config.get("NUM_WORKERS", 0)
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=modl_config["train"]["batch_size"],
            shuffle=False,
            num_workers=modl_config.get("NUM_WORKERS", 0)
        )

    def train(self, resume: bool = True):
        """训练模型"""
        self.model.train()
        best_loss = float('inf')
        start_epoch = 0

        modl_config = self.config["MoDL"]
        checkpoint_path = modl_config["train"]["checkpoint_path"]

        if resume and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            total_steps = checkpoint.get('total_steps', None)
            if total_steps is None:
                total_steps = len(self.train_loader) * modl_config["train"]["epochs"]

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=modl_config["train"]["learning_rate"] * 10,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True
            )
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch}")

        if self.scheduler is None:
            total_steps = len(self.train_loader) * modl_config["train"]["epochs"]
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=modl_config["train"]["learning_rate"] * 10,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True
            )

        for epoch in range(start_epoch, modl_config["train"]["epochs"]):
            total_loss = 0
            for batch_idx, (y, mask, x) in enumerate(self.train_loader):
                y = y.to(self.device, dtype=torch.complex64)
                mask = mask.to(self.device)
                x = x.to(self.device).unsqueeze(1)

                outputs = self.model(y, mask)
                loss = self.criterion(outputs, x.squeeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # 打印批次信息
                if batch_idx % 19 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f'Epoch: {epoch + 1}/{modl_config["train"]["epochs"]}, '
                          f'Batch: {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'LR: {current_lr:.2e}')

            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            print(f'Epoch {epoch + 1}/{modl_config["train"]["epochs"]}, '
                  f'Avg Loss: {avg_loss:.5f}, '
                  f'LR: {current_lr:.2e}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), modl_config["train"]["save_model_path"])
                print(f"Model saved to {modl_config['train']['save_model_path']}")

            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_loss': best_loss,
                'total_steps': total_steps,
            }, checkpoint_path)

        print("Training completed.")

    def evaluate(self):
        """评估模型

        返回:
            avg_psnr: 平均 PSNR
            avg_ssim: 平均 SSIM
        """
        self.model.eval()
        psnrs = []
        ssims = []
        # 存储每张图片的结果
        results = []

        with torch.no_grad():
            for y, mask, x in self.val_loader:
                y = y.to(self.device, dtype=torch.complex64)
                mask = mask.to(self.device)
                x = x.to(self.device).clamp(0, 1)
                output = self.model(y, mask).clamp(0, 1)

                batch_results = []
                for i in range(x.shape[0]):
                    # 从 y 中还原 undersampled image
                    undersampled_k = y[i].cpu().numpy()
                    un_img = np.abs(np.fft.ifft2(undersampled_k))  # k-space -> image

                    gt = x[i].cpu().numpy()
                    recon = output[i].cpu().numpy()
                    # PSNR
                    p = psnr(gt, recon, data_range=1.0)
                    psnrs.append(p)
                    # SSIM
                    s = ssim(gt, recon, data_range=1.0)
                    ssims.append(s)
                    # 存储结果
                    batch_results.append({
                        'ground_truth': gt,
                        'reconstructed': recon,
                        'noisy': un_img,
                        'psnr': p,
                        'ssim': s,
                        'index': f"{i}"
                        })

                results.extend(batch_results)

        return {
            'average_psnr': np.mean(psnrs),
            'average_ssim': np.mean(ssims),
            'results': results,
            'num_samples': len(results)
            }

    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def train_modl(config_path: str, resume: bool = True):
    """训练 MoDL 模型的主函数

    参数:
        config_path: JSON 配置文件路径
        resume: 是否从检查点恢复训练

    返回:
        model: 训练好的模型
        avg_psnr: 验证集平均 PSNR
        avg_ssim: 验证集平均 SSIM
    """
    trainer = MoDLTrainer(config_path)
    trainer.load_datasets()
    trainer.train(resume=resume)
    results = trainer.evaluate()
    print(f'Validation PSNR: {results["average_psnr"]:.2f} dB, SSIM: {results["average_ssim"]:.4f}')



if __name__ == "__main__":
    # 示例用法
    train_modl("../DL_params.json", resume=True)