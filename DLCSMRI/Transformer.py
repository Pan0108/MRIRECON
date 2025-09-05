# -*- coding: utf-8 -*-
"""
基于 CNN-Transformer 混合架构的压缩感知 MRI（CS-MRI）代码实现
重构版本：支持 JSON 配置 + 模块化设计
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
from einops import rearrange
from typing import Tuple


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path, model_name):
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        self.config = self.full_config.get(model_name, {})

        # 设置默认值
        defaults = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "BATCH_SIZE": 4,
            "NUM_WORKERS": 0,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-5,
            "NUM_EPOCHS": 500,
            "CHECKPOINT_PATH": f"./{model_name}_checkpoint.pth",
            "SAVE_MODEL_PATH": f"./{model_name}_model.pth",
            "OUTPUT_DIR": f"./output_{model_name}",
            "IMAGE_SIZE": 256,
            "DATA_PATHS": {
                "TRAIN_NOISY": "./data/train/undersampled",
                "TRAIN_MASK": "./data/train/mask",
                "TRAIN_CLEAN": "./data/train/full",
                "VAL_NOISY": "./data/val/undersampled",
                "VAL_MASK": "./data/val/mask",
                "VAL_CLEAN": "./data/val/full"
            },
            "model": {
                "patch_size": 16,
                "in_channels": 1,
                "embed_dim": 512,
                "num_heads": 16,
                "num_layers": 12,
                "num_residual_blocks": 8
            }
        }

        # 合并配置
        self.config = self._merge_dicts(defaults, self.config)

    def _merge_dicts(self, defaults, config):
        """递归合并字典"""
        for key, value in config.items():
            if isinstance(value, dict) and key in defaults:
                defaults[key] = self._merge_dicts(defaults[key], value)
            else:
                defaults[key] = value
        return defaults


class CustomCSMRIDataset(Dataset):
    """自定义 MRI 数据集 - 适配新配置结构"""

    def __init__(self, config, mode='train'):
        self.image_size = config["IMAGE_SIZE"]

        # 设置数据路径
        data_paths = config["DATA_PATHS"]
        if mode == 'train':
            self.kspace_dir = data_paths["TRAIN_NOISY"]
            self.mask_dir = data_paths["TRAIN_MASK"]
            self.full_dir = data_paths["TRAIN_CLEAN"]
        else:  # val
            self.kspace_dir = data_paths["VAL_NOISY"]
            self.mask_dir = data_paths["VAL_MASK"]
            self.full_dir = data_paths["VAL_CLEAN"]

        self.file_list = [f for f in os.listdir(self.full_dir) if f.endswith('.png')]
        self._validate_dirs()
        self._setup_data_pairs()

        # 定义转换
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def _validate_dirs(self):
        """验证目录是否存在"""
        required_dirs = {
            "undersampled": self.kspace_dir,
            "mask": self.mask_dir,
            "full": self.full_dir
        }
        missing = [name for name, path in required_dirs.items() if not os.path.exists(path)]
        if missing:
            raise ValueError(f"Missing required folders: {', '.join(missing)}")

    def _setup_data_pairs(self):
        """匹配数据文件"""
        self.data_pairs = []
        kspace_files = [f for f in os.listdir(self.kspace_dir) if f.endswith("_kspace.npy")]

        for kspace_file in kspace_files:
            base_name = kspace_file.replace("_kspace.npy", "")
            mask_file = f"{base_name}_mask.npy"

            if os.path.exists(os.path.join(self.mask_dir, mask_file)):
                self.data_pairs.append(base_name)
            else:
                print(f"Warning: Missing mask for {base_name}, skipping...")

        if not self.data_pairs:
            raise RuntimeError("No valid data pairs found. Check file naming and paths.")
        print(f"Found {len(self.data_pairs)} samples")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        base_name = self.data_pairs[idx]

        # 加载完整图像
        img_path = os.path.join(self.full_dir, f"{base_name}.png")
        img = Image.open(img_path).convert('L')
        x = self.transform(img).squeeze()

        # 加载 k-space
        kspace_path = os.path.join(self.kspace_dir, f"{base_name}_kspace.npy")
        kspace = np.load(kspace_path)
        y = torch.from_numpy(np.squeeze(kspace)).type(torch.complex64)

        # 加载掩码
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.npy")
        mask = torch.from_numpy(np.squeeze(np.load(mask_path))).bool()

        # 验证形状
        assert y.shape == (self.image_size, self.image_size), f"K-space shape mismatch: {y.shape}"
        assert mask.shape == (self.image_size, self.image_size), f"Mask shape mismatch: {mask.shape}"
        assert x.shape == (self.image_size, self.image_size), f"Image shape mismatch: {x.shape}"

        return y, mask, x


# ======================================================================================
# 模型组件 (保持不变)
# ======================================================================================
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        real = self.conv_re(x.real) - self.conv_im(x.imag)
        imag = self.conv_re(x.imag) + self.conv_im(x.real)
        return torch.complex(real, imag)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, complex_input=False):
        super().__init__()
        if complex_input:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, complex_input=False):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, complex_input=complex_input)
        self.conv2 = ConvBlock(channels, channels, complex_input=complex_input)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b d h w -> b (h w) d')


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def _ff_block(self, x):
        return self.mlp(x)

class HybridTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["model"]
        image_size = config["IMAGE_SIZE"]
        patch_size = self.config["patch_size"]
        embed_dim = self.config["embed_dim"]

        self.final_feature_size = image_size // 4
        self.num_patches = (self.final_feature_size // patch_size) ** 2

        # CNN 提取器
        self.initial_conv = nn.Sequential(
            ConvBlock(2, 64, kernel_size=7, padding=3, complex_input=False),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, embed_dim, kernel_size=3, stride=2, padding=1)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(embed_dim) for _ in range(self.config["num_residual_blocks"])]
        )

        # Patch Embedding & Positional Encoding
        self.patch_embed = PatchEmbedding(patch_size, embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer 编码器
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(embed_dim, self.config["num_heads"], config["model"].get("mlp_ratio", 4.))
              for _ in range(self.config["num_layers"])]
        )

        # Decoder 上采样重建
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16), nn.GELU(),
            nn.ConvTranspose2d(16, self.config["in_channels"], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, y, mask):
        y_real = y.real.unsqueeze(1)
        y_imag = y.imag.unsqueeze(1)
        y = torch.cat([y_real, y_imag], dim=1)
        mask = mask.unsqueeze(1)
        y = y * mask

        x = self.initial_conv(y)
        x = self.residual_blocks(x)

        B, C, H, W = x.shape
        x_patches = self.patch_embed(x)
        pos_embed = self.pos_embed[:, :H * W // self.config["patch_size"] ** 2]
        x_patches = x_patches + pos_embed

        x_transformed = self.transformer_encoder(x_patches)
        h = w = int((x_transformed.shape[1]) ** 0.5)
        x_transformed = rearrange(x_transformed, 'b (h w) d -> b d h w', h=h, w=w)

        x_recon = self.decoder(x_transformed)
        return x_recon.squeeze(1)


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


class TransformerTrainer:
    """Transformer 训练器 - 基于 CNN-Transformer 混合架构的 CS-MRI 重构训练器"""

    def __init__(self, config_path: str, model_name: str = "Transformer"):
        """
        初始化训练器

        Args:
            config_path: JSON 配置文件路径
            model_name: 配置中的模型名称（对应 config.json 中的键）
        """
        # 加载配置文件
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        self.config = self.full_config.get(model_name, {})

        # 合并默认配置
        self._set_defaults()

        # 设备设置
        self.device = torch.device(self.config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = HybridTransformer(self.config).to(self.device)

        # 损失函数
        self.criterion = CombinedLoss()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config.get("WEIGHT_DECAY", 1e-5)
        )

        # 学习率调度器（延迟初始化）
        self.scheduler = None

        # 创建输出目录
        os.makedirs(self.config.get("OUTPUT_DIR", f"./output_{model_name}"), exist_ok=True)

    def _set_defaults(self):
        """设置默认配置值"""
        defaults = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "BATCH_SIZE": 4,
            "NUM_WORKERS": 0,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-5,
            "NUM_EPOCHS": 500,
            "CHECKPOINT_PATH": f"./{self.config.get('model_name', 'Transformer')}_checkpoint.pth",
            "SAVE_MODEL_PATH": f"./{self.config.get('model_name', 'Transformer')}_model.pth",
            "OUTPUT_DIR": f"./output_{self.config.get('model_name', 'Transformer')}",
            "IMAGE_SIZE": 256,
            "DATA_PATHS": {
                "TRAIN_NOISY": "./data/train/undersampled",
                "TRAIN_MASK": "./data/train/mask",
                "TRAIN_CLEAN": "./data/train/full",
                "VAL_NOISY": "./data/val/undersampled",
                "VAL_MASK": "./data/val/mask",
                "VAL_CLEAN": "./data/val/full"
            },
            "model": {
                "patch_size": 16,
                "in_channels": 1,
                "embed_dim": 512,
                "num_heads": 16,
                "num_layers": 12,
                "num_residual_blocks": 8
            }
        }
        # 递归合并
        for k, v in defaults.items():
            if k not in self.config:
                self.config[k] = v
            elif isinstance(v, dict) and isinstance(self.config[k], dict):
                self.config[k] = {**v, **self.config[k]}

    def load_datasets(self):
        """加载训练和验证数据集"""
        train_dataset = CustomCSMRIDataset(self.config, mode='train')
        val_dataset = CustomCSMRIDataset(self.config, mode='val')

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.config.get("NUM_WORKERS", 0),
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.config.get("NUM_WORKERS", 0)
        )
        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    def train(self, resume: bool = False):
        """训练模型"""
        self.model.train()
        best_loss = 0
        start_epoch = 0

        checkpoint_path = self.config["CHECKPOINT_PATH"]

        if resume and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # 重新初始化 scheduler
            total_steps = len(self.train_loader) * self.config["NUM_EPOCHS"]
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config["LEARNING_RATE"] * 10,
                total_steps=total_steps,
                pct_start=0.3
            )
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', 0.0)
            print(f"Resuming training from epoch {start_epoch}, best LOSS: {best_loss:.4f}")

        # 如果没有从检查点恢复，初始化 scheduler
        if self.scheduler is None:
            total_steps = len(self.train_loader) * self.config["NUM_EPOCHS"]
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config["LEARNING_RATE"] * 10,
                total_steps=total_steps,
                pct_start=0.3
            )

        print(f"Starting training for {self.config['NUM_EPOCHS']} epochs...")
        for epoch in range(start_epoch, self.config["NUM_EPOCHS"]):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (y, mask, x) in enumerate(self.train_loader):
                y, mask, x = y.to(self.device), mask.to(self.device), x.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(y, mask)
                loss = self.criterion(output, x)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                if batch_idx % 19 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"Epoch: {epoch+1}/{self.config['NUM_EPOCHS']}, "
                          f"Batch: {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.config["SAVE_MODEL_PATH"])
                print(f"At Epoch {epoch+1}, Best model saved with LOSS: {best_loss:.4f}")

            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)

        print("Training completed.")

    def evaluate(self):
        """评估模型在验证集上的性能

        Returns:
            avg_psnr: 平均 PSNR (dB)
            avg_ssim: 平均 SSIM
        """
        self.model.eval()
        psnrs, ssims = [], []
        # 存储每张图片的结果
        results = []
        with torch.no_grad():
            for y, mask, x in self.val_loader:
                y, mask, x = y.to(self.device), mask.to(self.device), x.to(self.device).clamp(0, 1)
                output = self.model(y, mask).clamp(0, 1)  # 限制输出范围

                batch_results = []
                for i in range(x.shape[0]):
                    #从 y 中还原 undersampled image
                    undersampled_k = y[i].cpu().numpy()
                    un_img = np.abs(np.fft.ifft2(undersampled_k))  # k-space -> image

                    gt = x[i].cpu().numpy()
                    recon = output[i].cpu().numpy()
                    #PSNR
                    p = psnr(gt, recon, data_range=1.0)
                    psnrs.append(p)
                    #SSIM
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

    def save_model(self, path: str = None):
        """保存模型权重"""
        save_path = path or self.config["SAVE_MODEL_PATH"]
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path: str):
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def train_transformer(config_path: str, model_name: str = "Transformer", resume: bool = False):
    """
    训练 Transformer 模型的主函数

    Args:
        config_path: JSON 配置文件路径
        model_name: 配置中模型的键名
        resume: 是否从检查点恢复训练

    Returns:
        model: 训练好的模型
        avg_psnr: 验证集平均 PSNR
        avg_ssim: 验证集平均 SSIM
    """
    trainer = TransformerTrainer(config_path, model_name)
    trainer.load_datasets()
    trainer.train(resume=resume)
    results = trainer.evaluate()
    print(f"Final Evaluation - PSNR: {results["average_psnr"]:.2f} dB, SSIM: {results["average_ssim"]:.4f}")



# ======================================================================================
# 主程序入口（可选）
# ======================================================================================
if __name__ == "__main__":
    # 示例：使用默认配置训练
    train_transformer("DL_params.json", model_name="Transformer", resume=True)