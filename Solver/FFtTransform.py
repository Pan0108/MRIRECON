# -*- coding: utf-8 -*-
import numpy as np
import glob
from PIL import Image
import os

def FFt_backward(ksp):
    """逆向FFT变换"""
    ksp = np.squeeze(ksp)
    if ksp.ndim != 2:
        raise ValueError(f"Expected 2D k-space data, got {ksp.ndim}D data with shape {ksp.shape}")
    ksp = np.fft.fftshift(ksp)
    im = np.fft.ifft2(ksp)
    im = abs(np.fft.ifftshift(im))
    return im


def normalization(x):
    """
        将输入的二维图像数组归一化到 [0, 1]，再转换为 uint8 格式 [0, 255]

        参数:
            x: 输入的二维 numpy 数组（可以是灰度图像或其他单通道数据）

        返回:
            x_out: 归一化并转换为 uint8 的数组
    """
  # 处理分母为零的情况（如全零数组）
    x = np.abs(x)
    if x.max() == x.min():
        x_norm = np.zeros_like(x, dtype=np.float32)  # 避免除以零
    else:
        x_norm = (x - x.min()) / (x.max() - x.min())  # 归一化到 [0, 1]

    x_out = (x_norm * 255).astype(np.uint8)  # 转换为 uint8
    return x_out

def process_single_kspace(file_path):
    """
    处理单个k-space文件
    """
    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        reconstructed = FFt_backward(kspace)

        return reconstructed

    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None

def process_wrapper_kspace(args):
    """
    多进程包装函数

    参数:
        args: 元组，包含 (file_path, mask, wavelet_params)
    """
    file_path = args
    try:
        return process_single_kspace(file_path)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def process_kspace_files(input_files, output_folder, save_npy=True):
    """
    处理多个k-space文件并保存结果

    参数:
        input_files: 单个文件路径（str）或文件路径列表（List[str]）
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
    """
    try:
        # 统一处理输入文件（支持单个文件路径或列表）
        if isinstance(input_files, str):
            file_paths = [input_files]  # 单个文件转为列表
        else:
            file_paths = list(input_files)  # 确保是列表

        # 检查文件是否存在
        valid_files = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                print(f"文件不存在: {file_path}")

        if not valid_files:
            print("没有有效的k-space文件")
            return False

        # 检查/创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        print(f"准备处理 {len(valid_files)} 个k-space文件")

        results = []
        success_count = 0

        for file_path in valid_files:
            try:
                # 直接调用处理函数
                result = process_wrapper_kspace((file_path))
                if result is not None:
                    reconstructed = result
                    base_name = os.path.basename(file_path).split('.')[0]

                    # 保存PNG图像
                    Image.fromarray(normalization(reconstructed)).save(
                        os.path.join(output_folder, f"{base_name}_reconstructed.png"))

                    # 保存npy格式
                    if save_npy:
                        np.save(os.path.join(output_folder, f"{base_name}_reconstructed.npy"), reconstructed)

                    print(f"已保存 {base_name} 的结果")
                    success_count += 1
                    results.append(result)

            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {str(e)}")
                results.append(None)

        print(f"处理完成，成功处理 {success_count}/{len(valid_files)} 个文件")
        return results

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


if __name__ == '__main__':
    input_folder = r"D:\潘立星\python_work\pythonProject2\CS_Python\algorithm_testing\data_1\underkspace\slice_000_0.5_kspace.npy"
    output_folder = r"C:\Users\lixing.pan\Desktop\data_1"
    process_kspace_files(input_folder,output_folder, save_npy=True)