# -*- coding: utf-8 -*-
import sys
import os
import shutil
import random


def split_images_into_parts(input_dir, output_dir, num_parts=12):
    """
    将文件夹中的图片随机分成指定份数。

    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出文件夹路径，用于存放分好的图片
    :param num_parts: 分成的份数，默认为 12
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有图片文件
    image_files = [f for f in os.listdir(input_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    total_images = len(image_files)

    if total_images == 0:
        raise ValueError("输入目录中没有找到图片文件")

    print(f"Total images found: {total_images}")

    # 打乱图片顺序
    random.shuffle(image_files)

    # 计算每份的图片数量
    images_per_part = total_images // num_parts
    remainder = total_images % num_parts  # 剩余的图片分配到前几份

    # 创建子文件夹并分配图片
    start_idx = 0
    for i in range(num_parts):
        # 创建子文件夹
        part_dir = os.path.join(output_dir, f"part_{i + 1}")
        os.makedirs(part_dir, exist_ok=True)

        # 计算当前份的图片数量
        end_idx = start_idx + images_per_part + (1 if i < remainder else 0)

        # 获取当前份的图片
        current_part_images = image_files[start_idx:end_idx]

        # 将图片复制到对应的子文件夹
        for img_file in current_part_images:
            src_path = os.path.join(input_dir, img_file)
            dst_path = os.path.join(part_dir, img_file)
            shutil.copy(src_path, dst_path)

        print(f"Part {i + 1}: Copied {len(current_part_images)} images to {part_dir}")

        # 更新起始索引
        start_idx = end_idx


if __name__ == "__main__":
    pass