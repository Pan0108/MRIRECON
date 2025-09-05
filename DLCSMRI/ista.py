# -*- coding: utf-8 -*-
"""
ISTA-Net MRI 图像重建模型
支持从 JSON 加载配置、Trainer 类封装、断点续训、OneCycleLR、PSNR/SSIM 评估
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ========================================================
# 从 JSON 文件加载配置
# ========================================================
def load_config(json_path="DL_params.json", model_key="ISTA"):
    """加载 JSON 配置文件，并返回指定模型的配置"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        config_all = json.load(f)
    if model_key not in config_all:
        raise ValueError(f"Model '{model_key}' not found in config file. Available: {list(config_all.keys())}")
    return config_all[model_key]

# 加载 ISTA 配置
config = load_config("DL_params.json", "ISTA")


if not config:
    config = {
        "DEVICE": "cuda",
        "data": {
            "image_size": 256
        },
        "train": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "epochs": 1000,
            "save_model_path": "./ista_net_recon_model.pth",
            "checkpoint_path": "./ista_checkpoint.pth"
        },
        "optim": {
            "num_iters": 8,
            "channels": 256,
            "rho_init": 0.1
        },
        "NUM_WORKERS": 4,
        "WEIGHT_DECAY": 1e-4,
        "OUTPUT_DIR": "./output_ISTA",
        "DATA_PATHS": {
            "TRAIN_NOISY": "../data/train/undersampled",
            "TRAIN_MASK": "../data/train/mask",
            "TRAIN_CLEAN": "../data/train/full",
            "VAL_NOISY": "../data/val/undersampled",
            "VAL_MASK": "../data/val/mask",
            "VAL_CLEAN": "../data/val/full"
        }
    }

# 创建输出目录
os.makedirs(config["OUTPUT_DIR"], exist_ok=True)

# ========================================================
# 自定义数据集（使用 DATA_PATHS）
# ========================================================
class CustomCSMRIDataset(Dataset):
    def __init__(self, noisy_dir, mask_dir, clean_dir, image_size=256, transform=None):
        self.noisy_dir = noisy_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        self.file_list = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not self.file_list:
            raise RuntimeError(f"No image files found in {clean_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        base_name = os.path.splitext(img_name)[0]

        # 加载 clean 图像
        img_path = os.path.join(self.clean_dir, img_name)
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        x = torch.from_numpy(np.array(img)).float() / 255.0

        # 加载 k-space (noisy)
        kspace_path = os.path.join(self.noisy_dir, f"{base_name}_kspace.npy")
        if not os.path.exists(kspace_path):
            raise FileNotFoundError(f"K-space file not found: {kspace_path}")
        kspace = np.load(kspace_path)
        kspace = np.squeeze(kspace)
        y = torch.from_numpy(kspace).type(torch.complex64)

        # 加载 mask
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = np.load(mask_path)
        mask = np.squeeze(mask)
        mask = torch.tensor(mask, dtype=torch.bool)

        # 断言形状
        assert y.shape == (self.image_size, self.image_size), f"K-space shape mismatch: {y.shape}"
        assert mask.shape == (self.image_size, self.image_size), f"Mask shape mismatch: {mask.shape}"
        assert x.shape == (self.image_size, self.image_size), f"Image shape mismatch: {x.shape}"

        return y, mask, x


# ========================================================
# 网络结构（ISTA-Net）
# ========================================================
class SparseCoding(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.threshold = nn.ReLU()

    def forward(self, x):
        features = self.conv1(x)
        sparse_features = self.threshold(features)
        return self.conv2(sparse_features)


class DataConsistency(nn.Module):
    def __init__(self, rho_init=0.1):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(rho_init))

    def forward(self, u, y, mask):
        kspace_pred = fft.fft2(u.squeeze(1))
        residual = torch.where(mask, kspace_pred - y, 0)
        grad = fft.ifft2(residual).real.unsqueeze(1)
        return u - self.rho * grad


class ISTABlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.sparse_coding = SparseCoding(channels)
        self.data_consistency = DataConsistency()

    def forward(self, u, y, mask):
        u_sparse = self.sparse_coding(u)
        u_dc = self.data_consistency(u_sparse, y, mask)
        return u_dc


class ISTANetRecon(nn.Module):
    def __init__(self, num_iters=8, channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([ISTABlock(channels) for _ in range(num_iters)])

    def forward(self, y, mask):
        u = fft.ifft2(y).real.unsqueeze(1)  # 初始重建
        for block in self.blocks:
            u = block(u, y, mask)
        return u.squeeze(1)


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, l1_weight=0.5, l2_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target) + self.l2_weight * self.l2(pred, target)

# ========================================================
# Trainer 类（封装训练与评估）
# ========================================================
class Trainer:
    def __init__(self, config_path, model_name="ISTA"):
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        self.config = self.full_config.get(model_name, {})

        # 设备
        self.device = torch.device(self.config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = ISTANetRecon(
            num_iters=self.config["optim"]["num_iters"],
            channels=self.config["optim"]["channels"]
        ).to(self.device)
        #损失函数
        self.criterion = CombinedLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            weight_decay=self.config.get("WEIGHT_DECAY", 1e-4)
        )
        self.scheduler = None
        self.start_epoch = 0
        self.best_loss = float('inf')

    def get_dataloaders(self):
        train_dataset = CustomCSMRIDataset(
            noisy_dir=self.config["DATA_PATHS"]["TRAIN_NOISY"],
            mask_dir=self.config["DATA_PATHS"]["TRAIN_MASK"],
            clean_dir=self.config["DATA_PATHS"]["TRAIN_CLEAN"],
            image_size=self.config["data"]["image_size"]
        )
        val_dataset = CustomCSMRIDataset(
            noisy_dir=self.config["DATA_PATHS"]["VAL_NOISY"],
            mask_dir=self.config["DATA_PATHS"]["VAL_MASK"],
            clean_dir=self.config["DATA_PATHS"]["VAL_CLEAN"],
            image_size=self.config["data"]["image_size"]
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config.get("NUM_WORKERS", 0),
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False,
            num_workers=self.config.get("NUM_WORKERS", 0),
            pin_memory=True
        )
        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    def load_checkpoint(self):
        ckpt_path = self.config["train"]["checkpoint_path"]
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resuming from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(state, self.config["train"]["checkpoint_path"])
        if is_best:
            torch.save(state, self.config["train"]["save_model_path"])
            print(f"Best model saved to {self.config['train']['save_model_path']}")

    def train(self):
        
        num_epochs = self.config["train"]["epochs"]
        total_steps = len(self.train_loader) * num_epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["train"]["learning_rate"] * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True
        )

        # 如果有断点，加载
        self.load_checkpoint()

        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            total_loss = 0.0

            for y, mask, x in self.train_loader:
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

            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}, LR: {current_lr:.2e}')

            # 保存最佳模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存断点
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch)

        print("Training completed.")

    def evaluate(self):
        self.model.eval()
        psnrs = []
        ssims = []
        results = []  # 存储每张图片的结果
        with torch.no_grad():
            for y, mask, x in self.val_loader:
                y = y.to(self.device, dtype=torch.complex64)
                mask = mask.to(self.device)
                x = x.to(self.device)

                outputs = self.model(y, mask).clamp(0, 1).cpu().numpy()
                x = x.clamp(0, 1).cpu().numpy()

                batch_results = []
                for i in range(x.shape[0]):
                    # 从 y 中还原 undersampled image
                    undersampled_k = y[i].cpu().numpy()
                    un_img = np.abs(np.fft.ifft2(undersampled_k))  # k-space -> image

                    # PSNR
                    p = psnr(x[i], outputs[i],data_range=1.0)
                    psnrs.append(p)
                    # SSIM
                    s = ssim(x[i], outputs[i], data_range=1.0)
                    ssims.append(s)

                    # 存储结果
                    batch_results.append({
                        'ground_truth': x[i],
                        'reconstructed': outputs[i],
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


# ========================================================
# 主程序入口
# ========================================================
if __name__ == "__main__":
    
    config = "DL_params.json"
    # 初始化 Trainer
    trainer = Trainer(config)
    #load dataset
    trainer.get_dataloaders()
    # 训练
    trainer.train()
    # 验证评估
    results = trainer.evaluate()
    print(f'Validation PSNR: {results['average_psnr']:.2f} dB, SSIM: {results['average_ssim']:.4f}')