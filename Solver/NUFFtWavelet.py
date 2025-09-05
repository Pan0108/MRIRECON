# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pynufft as nf
import pywt
import numpy as np
import glob
from PIL import Image
import os
# # 生成径向采样轨迹

def generate_radial_trajectory(shape, num_rays=128, undersampling=0.75):
    """改进：确保轨迹点连续覆盖全范围"""
    height, width = shape
    center_y, center_x = height / 2, width / 2
    max_radius = np.sqrt((height / 2) ** 2 + (width / 2) ** 2)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    coords = []
    for angle in angles:
        dir_x, dir_y = np.cos(angle), np.sin(angle)
        # 增加采样密度
        steps = np.linspace(-max_radius, max_radius, int(max_radius * undersampling))
        for step in steps:
            kx = center_x + step * dir_x
            ky = center_y + step * dir_y
            if 0 <= kx < width and 0 <= ky < height:
                coords.append((kx - center_x, ky - center_y))
    # 归一化到[-0.5, 0.5]
    kspace_coords = np.array(coords, dtype=np.float32)
    kspace_coords[:, 0] = kspace_coords[:, 0] * (1 / width)
    kspace_coords[:, 1] = kspace_coords[:, 1] * (1 / height)

    return kspace_coords


# 小波软阈值函数
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# 基于小波和ISTA的压缩感知重建
def cs_recon_ista(kspace, traj, image_size=256, eps=0.01, tol=1e-5, L=0.01, lamda=0.1, max_iter=80):
    """
     #L = 0.01 (Lipschitz常数的倒数)
    """
    # 初始化NUFFT对象,这里支持长宽相等的图像
    nufft_obj = nf.NUFFT()
    Kd = (2 * image_size, 2 * image_size)  # 过采样size
    shape=(image_size,image_size)
    nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))

    # 初始化
    image_shape = (image_size,image_size)
    x = np.zeros(image_shape, dtype=np.complex128)

    # 标准化数据
    kspace_norm = kspace / np.max(np.abs(kspace))

    # 添加数值稳定性措施
    prev_error = np.inf

    for i in range(max_iter):
        # 计算梯度: A^H(Ax - y)
        residual = nufft_obj.forward(x) - kspace_norm
        gradient = nufft_obj.adjoint(residual)

        # 梯度下降步
        x = x - (L) * gradient

        # 使用更稳定的小波模式
        coeffs = pywt.wavedec2(np.real(x), 'sym8', level=2, mode='symmetric')

        # 软阈值处理
        coeffs_thresh = [coeffs[0]]
        for c in coeffs[1:]:
            c_thresh = [soft_threshold(arr, lamda * L) for arr in c]
            coeffs_thresh.append(tuple(c_thresh))

        # 反小波变换
        x = pywt.waverec2(coeffs_thresh, 'sym8', mode='symmetric')

        # 计算当前误差
        current_error = np.linalg.norm(residual)
        if i % 5 == 0:
            print(f'Iteration {i}/{max_iter}, Error: {current_error:.4f}')

        # 检查收敛性
        if abs(prev_error - current_error) < tol:
            print(f"Converged at iteration {i}")
            break
        elif current_error < eps:
            break
        if current_error > prev_error:
            if i/max_iter > 0.6:
                print(f"当前已迭代了{i}次，完成度大于{100*i/max_iter}%，可获得较好重建结果")
                break
            else:
                print(f"当前已迭代了{i}次，完成度小于{100 * i / max_iter}%; 梯度严重爆炸，参数L={L}，请向下微调该参数！")
                break

        prev_error = current_error

    return x


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

def process_single_kspace(file_path, mask, nufft_params=None):
    """
    处理单个k-space文件

    参数:
        kspace,  #降采样k空间数据
        nufft_obj,   #降采样的坐标轨迹
        image_size=256,  #图像的尺寸，默认height = width = image_size
        eps=0.01,  #图像迭代前后l2差值
        tol=1e-5,   #收敛容差
        L=0.01,    # (Lipschitz常数的倒数)
        lamda=0.1,   #平衡L1的参数，
        max_iter=80   #最大迭代次数
    """
    # 设置默认参数
    default_params = {
        "image_size": 256,
        "eps": 0.01,
        "tol": 0.00001,
        "L": 0.01,
        "lamda": 0.1,
        "max_iter": 80
    }

    # 合并默认参数和传入参数
    if nufft_params is None:
        nufft_params = default_params
    else:
        nufft_params = {**default_params, **nufft_params}

    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        reconstructed = cs_recon_ista(kspace, mask,
                                      image_size=nufft_params["image_size"],
                                      eps=nufft_params["eps"],
                                      tol= nufft_params["tol"],
                                      L=nufft_params["L"],
                                      lamda= nufft_params["lamda"],
                                      max_iter=nufft_params["max_iter"]
                                      )


        return  np.abs(reconstructed)

    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None

def process_wrapper_kspace(args):
    """
    多进程包装函数

    参数:
        args: 元组，包含 (file_path, mask, wavelet_params)
    """
    file_path, mask, nufft_params = args
    try:
        return process_single_kspace(file_path, mask, nufft_params)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def process_kspace_files(kspace_paths, mask_path, output_folder, save_npy=True, nufft_params=None):
    """
    处理多个k-space文件并保存结果

    参数:
        kspace_paths: k-space .npy文件路径列表
        mask_path: mask文件的路径
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
        nufft_params: NUFFT参数字典
    """

    try:
        # 检查/创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 加载mask
        mask = np.load(mask_path)
        print(f"加载mask文件: {mask_path}, 形状: {mask.shape}")

        # 检查是否有k-space文件
        if not kspace_paths:
            print("没有提供k-space文件路径")
            return False

        print(f"准备处理 {len(kspace_paths)} 个k-space文件")

        results = []
        success_count = 0

        # 逐个处理k-space文件
        for file_path in kspace_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    results.append(None)
                    continue

                # 检查文件扩展名
                if not file_path.lower().endswith('.npy'):
                    print(f"文件 {file_path} 不是.npy格式")
                    results.append(None)
                    continue

                # 直接调用处理函数
                result = process_wrapper_kspace((file_path, mask, nufft_params))
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

        print(f"处理完成，成功处理 {success_count}/{len(kspace_paths)} 个文件")
        return results

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

# 主程序
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tkinter as tk
    from tkinter import filedialog
    # 用户界面选择文件夹和文件
    def select_folder(title):
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        folder_path = filedialog.askdirectory(title=title)
        return folder_path


    def select_file(title):
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(title=title, filetypes=[("Numpy files", "*.npy")])
        return file_path

    # 用户选择输入文件夹
    input_folder = select_folder("选择k-space数据文件夹")
    if not input_folder:
        print("未选择输入文件夹，程序退出")


    # 用户选择mask文件
    mask_path = select_file("选择mask文件")
    if not mask_path:
        print("未选择mask文件，程序退出")

    # 用户选择输出文件夹
    output_folder = select_folder("选择输出文件夹")
    if not output_folder:
        print("未选择输出文件夹，程序退出")


    process_kspace_files(input_folder, mask_path, output_folder, save_npy=True, nufft_params=None)


   #画图显示


