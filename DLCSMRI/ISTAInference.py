# -*- coding: utf-8 -*-
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import numpy as np
import pynufft as nf
import os
from PIL import Image
from PyQt5.QtWidgets import QMessageBox

class CustomCSMRIDataset(Dataset):
    """Dataset for MRI , using interactive folder input."""

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

        # 加载 k-space
        kspace = np.load(kspace_path)
        kspace = np.squeeze(kspace)
        # 检查是否是 1D 数据 (Non-Cartesian)
        if kspace.ndim == 1:
            # 弹出警告窗口
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Non-Cartesian data detected!")
            msg.setInformativeText(f"File {base_name} is 1D (Non-Cartesian). Please use UNet for reconstruction.")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return False  # 返回 None 或跳过该样本

        y = torch.from_numpy(kspace).type(torch.complex64)

        # 加载掩码
        mask = np.load(mask_path)
        mask = np.squeeze(mask)
        mask = torch.tensor(mask, dtype=torch.bool)
        print(f"Input kspace shape: {y.shape}, mask shape: {mask.shape}")

        assert y.shape == (self.image_size, self.image_size)
        assert mask.shape == (self.image_size, self.image_size)


        return y, mask, base_name  # y: (H,W), mask: (H,W)

class InferenceConfigLoader:
    """Load model configuration from JSON file"""

    def __init__(self, config_path: str, model_name: str = "ISTA"):
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
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")

    return model


if __name__ == '__main__':
    # test_dataset.py
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    import traceback

    def test_dataset_safely(root_dir, image_size=256, max_samples=5):
        print("开始测试 CustomCSMRIDataset...")

        try:
            dataset = CustomCSMRIDataset(root_dir=root_dir, image_size=image_size)
        except Exception as e:
            print(f" 数据集初始化失败: {e}")
            return

        print(f"数据集初始化成功，共 {len(dataset)} 个样本")

        # 测试前几个样本
        for i in range(min(max_samples, len(dataset))):
            print(f"\n测试第 {i + 1} 个样本...")
            try:
                y, mask, base_name = dataset[i]
                print(f"   名称: {base_name}")
                print(f"   kspace shape: {y.shape}, dtype: {y.dtype}")
                print(f"   mask shape: {mask.shape}, dtype: {mask.dtype}")
                assert y.shape == (image_size, image_size), f"kspace shape 错误: {y.shape}"
                assert mask.shape == (image_size, image_size), f"mask shape 错误: {mask.shape}"
                print("样本加载正常")
            except Exception as e:
                print(f" 加载样本失败: {e}")

                traceback.print_exc()

        # 可选：测试 DataLoader


        print("\n测试 DataLoader...")
        try:
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
            for batch_idx, (y, mask, names) in enumerate(loader):
                if batch_idx >= 2:
                    break
                print(f"   Batch {batch_idx}: y.shape={y.shape}, mask.shape={mask.shape}, names={names}")
            print("DataLoader 正常运行")
        except Exception as e:
            print(f"DataLoader 出错: {e}")
            traceback.print_exc()


    root_dir = r"C:\Users\lixing.pan\Desktop\data_1\val"  # 实际路径
    if os.path.exists(root_dir):
        test_dataset_safely(root_dir)
    else:
        print(f"路径不存在: {root_dir}")