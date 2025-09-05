# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pynufft as nf
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from typing import List, Tuple, Dict, Any
import torch.nn.functional as F
import torchvision
from pytorch_msssim import SSIM

class ConfigLoader:
    """Load model configuration from JSON file"""

    def __init__(self, config_path: str, model_name: str = "U-Net"):
        self.config_path = config_path
        self.model_name = model_name
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                data = json.load(file)

                if self.model_name not in data:
                    raise ValueError(f"Model {self.model_name} not found in configuration file")

                config = data[self.model_name]

                # Convert device string to torch.device
                device_str = config.get("DEVICE", "cuda").lower()
                config["DEVICE"] = torch.device(
                    "cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu"
                )

                return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {self.config_path} not found!")
        except json.JSONDecodeError:
            raise ValueError("Error decoding JSON file!")
        except Exception as e:
            raise RuntimeError(f"Error loading parameters: {str(e)}")


class HybridLoss(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.mse_loss = nn.MSELoss()
        # 频域损失（确保k空间一致性）
        self.fft_loss = nn.L1Loss()
       
        try:
            self.vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        except Exception as e:
            print(f"Failed to download VGG16 weights: {e}")
            print("Attempting to use locally cached version...")
            self.vgg = torchvision.models.vgg16(weights=None).features[:16]
       
        for param in self.vgg.parameters():
            param.requires_grad = False

        # 将 VGG 模型移动到指定设备
        self.vgg.to(device)

    def forward(self, pred, target):
        # 添加安全截断
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        # L1损失
        l1_loss = self.l1(pred, target)
        #l2
        l2_loss = self.mse_loss(pred, target)

        # SSIM损失
        ssim_loss = 1 - self.ssim(pred, target)  # SSIMLoss返回的是1-SSIM
        
   
        # 总损失（根据权重组合）
        total_loss = 0.7 * l1_loss + 0.2 * l2_loss + 0.1 * ssim_loss

        # 返回总损失和各分量损失（用于监控）
        return total_loss, {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "l2": l2_loss.item(),
            "ssim": ssim_loss.item(),
            # "fft": fft_l1.item()
        }


class UNet(nn.Module):
    """Improved U-Net architecture with skip connections"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        super().__init__()
        # Encoder (4 downsampling layers)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        # Middle layer
        self.bottleneck = self._block(512, 1024)
        # Decoder (4 upsampling layers + skip connections)
        self.dec4 = self._block(1024 + 512, 512)  # Skip connection concatenation
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Middle layer
        bn = self.bottleneck(self.pool(e4))
        # Decoder (with skip connections)
        d4 = self.dec4(torch.cat([self.upsample(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        return self.final(d1)


class DenoisingDataset(Dataset):
    """Dataset for MRI denoising with NUFFT support"""

    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        """
        Initialize dataset for training or validation

        Args:
            config: Loaded configuration dictionary
            mode: Either 'train' or 'val'
        """
        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"

        prefix = "TRAIN" if mode == "train" else "VAL"
        self.noisy_dir = config["DATA_PATHS"][f"{prefix}_NOISY"]
        self.mask_dir = config["DATA_PATHS"][f"{prefix}_MASK"]
        self.clean_dir = config["DATA_PATHS"][f"{prefix}_CLEAN"]
        self.image_size = config.get("IMAGE_SIZE", 256)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self._setup_data_pairs()

    def _setup_data_pairs(self):
        """Validate and setup data paths"""
        self.data_pairs = []

        kspace_files = [
            f for f in os.listdir(self.noisy_dir)
            if f.endswith("_kspace.npy") and not f.startswith('.')
        ]

        for kspace_file in kspace_files:
            base_name = kspace_file.replace("_kspace.npy", "")
            kspace_path = os.path.join(self.noisy_dir, kspace_file)
            mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.npy")
            clean_path = os.path.join(self.clean_dir, f"{base_name}.png")

            if all(os.path.exists(p) for p in [mask_path, clean_path]):
                self.data_pairs.append((kspace_path, mask_path, clean_path))

        if len(self.data_pairs) == 0:
            raise RuntimeError("No valid data triplets found. Check paths.")
        # print(f"Found {len(self.data_pairs)} samples.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        kspace_path, mask_path, clean_path = self.data_pairs[idx]
        base_name = os.path.basename(kspace_path).replace("_kspace.npy", "")
        # Load k-space data
        kspace_noisy = np.load(kspace_path)

        # 处理 Cartesian 数据 (1, H, W)
        if kspace_noisy.ndim == 3 and kspace_noisy.shape[0] == 1:
            # print(f"[{base_name}] Using Cartesian reconstruction")
            kspace_noisy = kspace_noisy.squeeze(0)
            # 对于 Cartesian，mask 是 2D 二值矩阵，不是轨迹
            mask = np.load(mask_path)
            # print(f"[{base_name}] mask shape: {mask.shape}")

            # 应用 mask（如果是欠采样）
            kspace_noisy = kspace_noisy * mask

            # 重建
            image_complex = np.fft.ifft2(kspace_noisy)
            # 处理 2D Cartesian 数据 (H, W)
        elif kspace_noisy.ndim == 2:
            # print(f"[{base_name}] Using Cartesian reconstruction (2D)")
            # 可能也需要 mask
            try:
                mask = np.load(mask_path)
                kspace_noisy = kspace_noisy * mask
            except:
                pass
            image_complex = np.fft.ifft2(kspace_noisy)

        # 处理 Non-Cartesian 数据 (N,)
        elif kspace_noisy.ndim == 1:
            # print(f"[{base_name}] Using NUFFT reconstruction")
            traj = np.load(mask_path)
            # print(f"[{base_name}] traj shape: {traj.shape}")

            # 确保 traj 是 (N, 2)
            if traj.ndim == 1 or traj.shape[0] == 2:
                traj = traj.T if traj.shape[0] == 2 else traj.reshape(-1, 2)

            # 归一化到 [-0.5, 0.5]
            if traj.max() > 1.0:
                traj = (traj - 128) / 256

            shape = (self.image_size, self.image_size)
            image_complex = self.NUFFt_backward(kspace_noisy, traj, shape)

        else:
            raise ValueError(f"Unsupported kspace dimension: {kspace_noisy.ndim}")

        # Convert to magnitude image and normalize
        noisy_image = np.abs(image_complex)
        noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min() + 1e-8)
        noisy_image = Image.fromarray((noisy_image * 255).astype(np.uint8)).convert("L")

        # Load clean target image
        clean_image = Image.open(clean_path).convert('L')

        # Apply transformations
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

    def NUFFt_backward(self,kspace, traj, shape=(256, 256)):
        """逆向NUFFT变换"""
        # 初始化NUFFT对象,这里支持长宽相等的图像
        nufft_obj = nf.NUFFT()
        Kd = (2 * shape[0], 2 * shape[1])  # 过采样size
        nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))
        # 重建
        im = nufft_obj.adjoint(kspace)
        return im

class MRIDenoisingTrainer:
    """Main training class that handles the complete pipeline"""

    def __init__(self, config_path: str, model_name: str = "U-Net"):
        self.config_loader = ConfigLoader(config_path, model_name)
        self.config = self.config_loader.config
        self.device = self.config["DEVICE"]

        # Initialize components
        self.model = self._init_model()
        self.criterion = HybridLoss(device=self.device)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        # Data loaders
        self.train_loader = self._init_data_loader("train")
        self.val_loader = self._init_data_loader("val")

    def _init_model(self) -> nn.Module:
        """Initialize model with config parameters"""
        in_channels = self.config.get("IN_CHANNELS", 1)
        out_channels = self.config.get("OUT_CHANNELS", 1)
        return UNet(in_channels, out_channels).to(self.device)

    def _init_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer with config parameters"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config["WEIGHT_DECAY"]
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["NUM_EPOCHS"],
            eta_min=self.config["LEARNING_RATE"] * 0.01
        )

    def _init_data_loader(self, mode: str) -> DataLoader:
        """Initialize data loader for train/val"""
        dataset = DenoisingDataset(self.config, mode)
        return DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=(mode == "train"),
            num_workers=self.config["NUM_WORKERS"],
            pin_memory=True,
            persistent_workers=True
        )

    def train(self):
        """Main training loop"""
        best_loss = float('inf')
        start_epoch = 0

        # 恢复训练（如果启用且检查点存在）
        if self.config["RESUME_TRAINING"] and os.path.exists(self.config["CHECKPOINT_PATH"]):
            start_epoch, best_loss = self._load_checkpoint()

        for epoch in range(start_epoch, self.config["NUM_EPOCHS"]):
            self.model.train()
            epoch_losses = {
                "total": 0.0,
                "l1": 0.0,
                "l2": 0.0,
                "ssim": 0.0,
            }
            
            batch_count = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                total_loss, loss_components = self.criterion(outputs, targets)
                
                total_loss.backward()
                self.optimizer.step()
                
                # 累加各项损失
                batch_size = inputs.size(0)
                for key in epoch_losses:
                    epoch_losses[key] += loss_components[key] * batch_size
                
                batch_count += batch_size

            # 计算平均损失
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
            
            # 学习率更新
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印详细损失信息
            print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}, LR: {current_lr:.2e}, Total Loss: {epoch_losses['total']:.5f}")
            print(f"\tL1 Loss: {epoch_losses['l1']:.4f} (70%: {0.7*epoch_losses['l1']:.4f})")
            print(f"  \tL2 Loss: {epoch_losses['l2']:.4f} (20%: {0.2*epoch_losses['l2']:.4f})")
            print(f"\tSSIM Loss: {epoch_losses['ssim']:.4f} (10%: {0.1*epoch_losses['ssim']:.4f})")
            
            
            # 保存最佳模型
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                self._save_checkpoint(epoch, best_loss)
                print(f"save to {self.config["CHECKPOINT_PATH"]} at Epoch {epoch+1}")


    
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        results = []  # 存储每张图片的结果

        with torch.no_grad():
            for idx, (noisy, clean) in enumerate(self.val_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                recon = self.model(noisy)

                batch_results = []
                for i in range(recon.shape[0]):
                    # 处理图像数据
                    gt_img_np = clean[i].cpu().numpy().squeeze()
                    recon_img_np = recon[i].cpu().numpy().squeeze()
                    noisy_img_np = noisy[i].cpu().numpy().squeeze()

                    # 归一化
                    gt_img_np = (gt_img_np - gt_img_np.min()) / (gt_img_np.max() - gt_img_np.min())
                    recon_img_np = (recon_img_np - recon_img_np.min()) / (recon_img_np.max() - recon_img_np.min())
                    noisy_img_np = (noisy_img_np - noisy_img_np.min()) / (noisy_img_np.max() - noisy_img_np.min())

                    # 计算PSNR和SSIM
                    current_psnr = psnr(gt_img_np, recon_img_np, data_range=1.0)
                    current_ssim = ssim(gt_img_np, recon_img_np, data_range=1.0)
                    total_psnr += current_psnr
                    total_ssim += current_ssim

                    # 存储结果
                    batch_results.append({
                        'ground_truth': gt_img_np,
                        'reconstructed': recon_img_np,
                        'noisy': noisy_img_np,
                        'psnr': current_psnr,
                        'ssim': current_ssim,
                        'index': f"{idx}_{i}"
                    })

                results.extend(batch_results)

        avg_psnr = total_psnr / len(results)
        avg_ssim = total_ssim / len(results)
        print(f"avg_psnr:{avg_psnr}\navg_ssim:{avg_ssim}")

        return {
            'average_psnr': avg_psnr,
            'average_ssim': avg_ssim,
            'results': results,
            'num_samples': len(results)
        }

    def _save_checkpoint(self, epoch: int, best_loss: float):
        """Save training checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss
        }
        torch.save(state, self.config["CHECKPOINT_PATH"])

    def _load_checkpoint(self) -> Tuple[int, float]:
        """Load training checkpoint"""
        checkpoint = torch.load(self.config["CHECKPOINT_PATH"], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['best_loss']


def main():
    parser = argparse.ArgumentParser(description="MRI Denoising Trainer")
    parser.add_argument("--config", type=str, default="./DL_params.json",
                        help="Path to configuration JSON file")
    parser.add_argument("--model", type=str, default="U-Net",
                        help="Model name in config file")
    args = parser.parse_args()

    # Initialize and run trainer
    trainer = MRIDenoisingTrainer(args.config, args.model)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()