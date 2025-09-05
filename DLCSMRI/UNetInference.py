# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import pynufft as nf
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any


class DenoisingDataset(Dataset):
    """Dataset for MRI denoising with NUFFT support, using interactive folder input."""

    def __init__(self, root_dir: str, image_size: int = 256):
        """
        Initialize dataset using a root directory selected interactively.

        Args:
            root_dir: Root directory containing 'undersampled', 'mask', and 'clean' subfolders.
            image_size: Target size for resizing images (default: 256)
        """
        self.root_dir = root_dir
        self.image_size = image_size

        # Define subdirectories
        self.undersampled_dir = os.path.join(root_dir, "undersampled")
        self.mask_dir = os.path.join(root_dir, "mask")


        # Validate directories
        required_dirs = {
            "undersampled": self.undersampled_dir,
            "mask": self.mask_dir,
        }
        missing = [name for name, path in required_dirs.items() if not os.path.exists(path)]
        if missing:
            raise ValueError(f"Missing required folders: {', '.join(missing)}")

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.data_pairs = []
        self._setup_data_pairs()

    def _NUFFt_backward(self,kspace, traj, shape=(256, 256)):
        """Inverse NUFFT reconstruction."""
        nufft_obj = nf.NUFFT()
        Kd = (2 * shape[0], 2 * shape[1])  # Oversampling
        try:
            nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))
            im = nufft_obj.adjoint(kspace)
            return np.abs(im)
        except Exception as e:
            raise RuntimeError(f"NUFFT reconstruction failed: {e}")

    def _setup_data_pairs(self):
        """Match kspace, mask, and clean image files by base name."""
        self.data_pairs = []

        kspace_files = [
            f for f in os.listdir(self.undersampled_dir)
            if f.endswith("_kspace.npy") and not f.startswith('.')
        ]

        for kspace_file in kspace_files:
            base_name = kspace_file.replace("_kspace.npy", "")
            kspace_path = os.path.join(self.undersampled_dir, kspace_file)
            mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.npy")


            if all(os.path.exists(p) for p in [mask_path, kspace_path]):
                self.data_pairs.append((kspace_path, mask_path))
            else:
                print(f"Warning: Missing files for {base_name}, skipping...")

        if len(self.data_pairs) == 0:
            raise RuntimeError("No valid data triplets found. Check file naming and paths.")
        print(f"Found {len(self.data_pairs)} samples in {self.root_dir}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        kspace_path, mask_path = self.data_pairs[idx]
        base_name = os.path.basename(kspace_path).replace("_kspace.npy", "")

        # Load k-space data
        kspace_noisy = np.load(kspace_path)
        print(f"[{base_name}] kspace shape: {kspace_noisy.shape}")

        # 处理 Cartesian 数据 (1, H, W)
        if kspace_noisy.ndim == 3 and kspace_noisy.shape[0] == 1:
            print(f"[{base_name}] Using Cartesian reconstruction")
            kspace_noisy = kspace_noisy.squeeze(0)
            # 对于 Cartesian，mask 是 2D 二值矩阵，不是轨迹
            mask = np.load(mask_path)
            print(f"[{base_name}] mask shape: {mask.shape}")

            # 应用 mask（如果是欠采样）
            kspace_noisy = kspace_noisy * mask

            # 重建
            image_complex = np.fft.ifft2(kspace_noisy)
            # 处理 2D Cartesian 数据 (H, W)
        elif kspace_noisy.ndim == 2:
            print(f"[{base_name}] Using Cartesian reconstruction (2D)")
            # 可能也需要 mask
            try:
                mask = np.load(mask_path)
                kspace_noisy = kspace_noisy * mask
            except:
                pass
            image_complex = np.fft.ifft2(kspace_noisy)

        # 处理 Non-Cartesian 数据 (N,)
        elif kspace_noisy.ndim == 1:
            print(f"[{base_name}] Using NUFFT reconstruction")
            traj = np.load(mask_path)
            print(f"[{base_name}] traj shape: {traj.shape};kspace shape{kspace_noisy.shape}")

            # 确保 traj 是 (N, 2)
            if traj.ndim == 1 or traj.shape[0] == 2:
                traj = traj.T if traj.shape[0] == 2 else traj.reshape(-1, 2)

            # 归一化到 [-0.5, 0.5]
            if traj.max() > 1.0:
                traj = (traj - 128) / 256


            shape = (self.image_size, self.image_size)
            image_complex = self._NUFFt_backward(kspace_noisy, traj, shape)

        else:
            raise ValueError(f"Unsupported kspace dimension: {kspace_noisy.ndim}")

        # Convert to magnitude image and normalize
        noisy_image = np.abs(image_complex)
        noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min() + 1e-8)
        noisy_image = Image.fromarray((noisy_image * 255).astype(np.uint8)).convert("L")


        # Apply transforms
        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, base_name  # 返回名称用于保存文件


class InferenceConfigLoader:
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


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.abs(image.squeeze())
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)


def load_model(model,config: dict, checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint"""

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config["DEVICE"])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")

    return model

