# -*- coding: utf-8 -*-

from mask_2d import *
import os
import numpy as np
from PIL import Image
import torch
import torch.fft as fft




# 生成欠采样 k-space 数据和对应的 mask 示例
def preprocess_data(input_dir, output_dir1, output_dir2,image_size=256):
    """
    导入全采样图片，生成欠采样 k-space 数据，并保存对应的 mask。
    :param input_dir: 输入图像目录（支持 .png, .jpg 等格式）
    :param output_dir: 输出目录（保存欠采样 k-space 和 mask）
    :param image_size: 统一调整图像大小为 (image_size, image_size)
    :param sampling_ratio: 欠采样比例
    """
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith(('.png', '.jpg')):  # 只处理图片文件
            continue

        # 加载并预处理图像
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('L').resize((image_size, image_size))  # 转灰度图并调整大小
        x = np.array(img) / 255.0  # 归一化到 [0, 1]
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # 转换为 Tensor 并增加通道维度

        # 生成 k-space 数据
        kspace = fft.fftshift(fft.fft2(x_tensor), dim=(-2, -1))  # 计算 k-space 并中心化

        # 生成 mask
        # mask = mask2d_phase((image_size, image_size), undersampling=0.4, radius=15,axis=1)
        # mask = mask2d_spiral((image_size, image_size), num_turns=50, undersampling=1, spacing_power=2)
        # mask = mask2d_center((image_size, image_size),center_r=15, undersampling=0.5)
        mask = mask2d_radial(256, num_rays=40, undersampling=1)
        # 应用 mask 进行欠采样
        undersampled_kspace = kspace.numpy() * mask

        # 保存欠采样 k-space 数据
        kspace_output_path = os.path.join(output_dir1, img_file.replace('.png', '_kspace.npy'))
        np.save(kspace_output_path, undersampled_kspace)

        # 保存 mask
        mask_output_path = os.path.join(output_dir2, img_file.replace('.png', '_mask.npy'))
        np.save(mask_output_path, mask)

        print(f"Processed {img_file}:")
        print(f"  Saved k-space to {kspace_output_path}")
        print(f"  Saved mask to {mask_output_path}")

# 主程序入口
if __name__ == "__main__":
    # 输入目录：存放全采样图片
    input_dir = r"C:\Users\lixing.pan\Desktop\data_1\val\full"

    # 输出目录：保存欠采样 k-space
    output_dir1 = r"C:\Users\lixing.pan\Desktop\data_1"
    # 输出目录：保存欠采样 mask
    output_dir2 = r"C:\Users\lixing.pan\Desktop\data_1"

    # 执行数据预处理
    preprocess_data(input_dir, output_dir1,output_dir2, image_size=256)# 图像统一调整大小