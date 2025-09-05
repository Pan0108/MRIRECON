# -*- coding: utf-8 -*-
import numpy as np
import glob
from PIL import Image
import os
import pynufft as nf

def NUFFt_backward(kspace,traj,shape=(256,256)):
    """逆向NUFFT变换"""
    # 初始化NUFFT对象,这里支持长宽相等的图像
    nufft_obj = nf.NUFFT()
    Kd = (2 * shape[0], 2 * shape[1])  # 过采样size
    nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))
    #重建
    im = normalization(np.abs(nufft_obj.adjoint(kspace)))
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


def process_single_kspace(file_path,traj):
    """
    处理单个k-space文件
    """
    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        reconstructed = NUFFt_backward(kspace, traj)

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
    file_path, traj = args
    try:
        return process_single_kspace(file_path,traj)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def process_kspace_files(file_paths, traj_path, output_folder, save_npy=True):
    """
    处理k-space文件并保存结果

    参数:
        file_paths: 包含k-space .npy文件路径的列表
        traj_path: mask文件的路径
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
    """

    try:
        # 检查/创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 加载mask
        mask = np.load(traj_path)
        print(f"加载mask文件: {traj_path}, 形状: {mask.shape}")

        print(f"找到 {len(file_paths)} 个k-space文件")

        if not file_paths:
            print("没有找到k-space文件")
            return False

        results = []
        success_count = 0

        # 处理每个文件
        for file_path in file_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    results.append(None)
                    continue

                # 检查文件扩展名
                if not file_path.lower().endswith('.npy'):
                    print(f"跳过非npy文件: {file_path}")
                    results.append(None)
                    continue

                # 直接调用处理函数
                result = process_wrapper_kspace((file_path, mask))
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

        print(f"处理完成，成功处理 {success_count}/{len(file_paths)} 个文件")
        return results

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

if __name__ == '__main__':
    input_folder = r"C:\Users\lixing.pan\Desktop\data_1\nufft\underkspace"
    traj_path = r"C:\Users\lixing.pan\Desktop\data_1\nufft\traj\traj_0.npy"
    output_folder = r"C:\Users\lixing.pan\Desktop\data_1\nufft"
    process_kspace_files(input_folder, traj_path, output_folder, save_npy=True)