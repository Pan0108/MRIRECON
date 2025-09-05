# -*- coding: utf-8 -*-
import numpy as np
import pywt
import os
import glob
import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from numba import jit, config
import pyfftw

# 配置Numba
config.THREADING_LAYER = 'threadsafe'

# 启用PyFFTW缓存
pyfftw.interfaces.cache.enable()

# 小波变换
def sparse_wavelet_transform(x, threshold, wavelet='db2', level=2, axes=(0, 1)):
    coeffs = pywt.wavedec2(x, wavelet, level=level, axes=axes)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs, padding=0, axes=axes)
    arr_thresholded = np.where(np.abs(arr) < threshold, 0, arr - threshold * np.sign(arr))
    return arr_thresholded, coeff_slices

# 逆变换
def inverse_sparse_wavelet_transform(arr_thresholded, coeff_slices, shape, wavelet='haar', axes=(0, 1)):
    coeffs_from_arr = pywt.array_to_coeffs(arr_thresholded, coeff_slices)
    im = pywt.waverecn(coeffs_from_arr, wavelet, axes=axes)
    for i in range(len(axes)):
        if shape[i] < im.shape[i]:  # 如果形状不匹配，删除最后一行
            im = np.delete(im, (im.shape[i] - 1), axis=i)
    return im
# 维度匹配
def dim_match(A_shape, B_shape):
    A_out_shape = A_shape
    B_out_shape = B_shape
    if len(A_shape) < len(B_shape):
        for _ in range(len(A_shape), len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape), len(A_shape)):
            B_out_shape += (1,)
    return A_out_shape, B_out_shape

# 使用PyFFTW加速的FFT
def FFt_forward(im, mask, axes=(0, 1)):
    im = np.fft.fftshift(im, axes)
    ksp = pyfftw.interfaces.numpy_fft.fft2(im, axes=axes)
    ksp = np.fft.ifftshift(ksp, axes)
    if len(ksp.shape) != len(mask.shape):
        ksp_out_shape, mask_out_shape = dim_match(ksp.shape, mask.shape)
        mksp = np.multiply(ksp.reshape(ksp_out_shape), mask.reshape(mask_out_shape))  # apply mask
    else:
        mksp = np.multiply(ksp, mask)  # apply mask
    return mksp

# 使用PyFFTW加速的逆FFT
def FFt_backward(ksp, axes=(0, 1)):
    ksp = np.fft.fftshift(ksp, axes)
    im = pyfftw.interfaces.numpy_fft.ifft2(ksp, axes=axes)
    im = np.fft.ifftshift(im, axes)
    return im

def normalization(x):
    """将输入的二维图像数组归一化到 [0, 1]，再转换为 uint8 格式 [0, 255]"""
    # 处理分母为零的情况（如全零数组）
    if x.max() == x.min():
        x_norm = np.zeros_like(x, dtype=np.float32)  # 避免除以零
    else:
        x_norm = (x - x.min()) / (x.max() - x.min())  # 归一化到 [0, 1]

    x_out = (x_norm * 255).astype(np.uint8)  # 转换为 uint8
    return x_out


def run_test(b, mask, epsilon=1e-4, n_max=50, tol=1e-4, decfac=0.5, threshold=5, wavelet='haar', level=4):
    '''
    argmin ||A(x)-b|||_2^2 + lambda_val*||DWt(x)||_1
    :param epsilon: 1e-6
    :param n_max: 100
    :param tol: 1e-4
    :param decfac: 0.5
    :param threshold: 2
    :param wavelet: haar
    :param level: 4
    :return: x1
    '''
    scaling = max(np.abs(FFt_backward(b).flatten()))  # 取b傅里叶变换之后的最大值，作归一化
    # 标准化处理
    y = b / scaling
    x0 = FFt_backward(y)
    x = x0.copy()

    lambda_init = np.max(np.abs(x0))
    lambda_val = lambda_init
    diff_k_space = FFt_forward(x0, mask)
    f_current = np.linalg.norm(diff_k_space) + lambda_val * np.sum(np.abs(x0))  # 计算初始功能函数


    # MM迭代重建
    while lambda_val > lambda_init * tol:
        for n in range(n_max):
            f_previous = f_current
            # 小波变换，软阈值，逆变换
            arr_thresholded, coeff_slices = sparse_wavelet_transform(
                x, threshold, wavelet=wavelet, level=level, axes=(0, 1))

            x_reconstructed_iter = inverse_sparse_wavelet_transform(
                arr_thresholded, coeff_slices, shape=x0.shape, wavelet=wavelet, axes=(0, 1))

            # 求残差，然后迭代更新
            Ksp_x_reconstructed_iter = FFt_forward(x_reconstructed_iter, mask)
            diff_k_space = Ksp_x_reconstructed_iter - y
            diff_image = FFt_backward(diff_k_space)

            # 更新图像
            x = x.astype(dtype=np.complex64)
            x -= diff_image / 1.1

            # 计算新的L1,L2范数
            f_current = np.linalg.norm(diff_k_space) + lambda_val * np.sum(np.abs(x_reconstructed_iter))

            # 检测收敛
            if np.linalg.norm(f_current - f_previous) / np.linalg.norm(f_current + f_previous) < tol:
                break

        # 迭代停止条件判断
        if np.linalg.norm(diff_k_space) < epsilon:
            print(f'在λ={lambda_val}时达到精度要求')
            break

        lambda_val *= decfac

    print("最终l1权重因子:", lambda_val)
    x1 = abs(x_reconstructed_iter)

    return x1


# 处理单个k-space文件
def process_single_kspace(file_path, mask, wavelet_params=None):
    """
    处理单个k-space文件

    参数:
        file_path: k-space文件路径
        mask: 掩码数据
        wavelet_params: 小波参数字典，包含以下键:
            - epsilon: 收敛阈值
            - n_max: 最大迭代次数
            - tol: 容差
            - decfac: 衰减因子
            - threshold: 小波阈值
            - wavelet: 小波类型
            - level: 小波分解层数
    """
    # 设置默认参数
    default_params = {
        "epsilon": 1e-4,
        "n_max": 50,
        "tol": 1e-4,
        "decfac": 0.5,
        "threshold": 5,
        "wavelet": 'haar',
        "level": 2
    }

    # 合并默认参数和传入参数
    if wavelet_params is None:
        wavelet_params = default_params
    else:
        wavelet_params = {**default_params, **wavelet_params}

    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        reconstructed = run_test(
            kspace, mask,
            epsilon=wavelet_params["epsilon"],
            n_max=wavelet_params["n_max"],
            tol=wavelet_params["tol"],
            decfac=wavelet_params["decfac"],
            threshold=wavelet_params["threshold"],
            wavelet=wavelet_params["wavelet"],
            level=wavelet_params["level"]
        )

        return  reconstructed

    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None



def process_wrapper_kspace(args):
    """
    多进程包装函数

    参数:
        args: 元组，包含 (file_path, mask, wavelet_params)
    """
    file_path, mask, wavelet_params = args
    try:
        return process_single_kspace(file_path, mask, wavelet_params)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


import os
import numpy as np
from PIL import Image
import multiprocessing

def process_kspace_files(kspace_paths, mask_path, output_folder, save_npy=True, wavelet_params=None):
    """
    处理多个k-space文件并保存结果

    参数:
        kspace_paths: k-space .npy文件路径列表
        mask_path: mask文件的路径
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
        wavelet_params: 小波变换参数字典
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

        # 过滤出有效的.npy文件
        valid_paths = []
        for path in kspace_paths:
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                continue
            if not path.lower().endswith('.npy'):
                print(f"文件 {path} 不是.npy格式")
                continue
            valid_paths.append(path)

        if not valid_paths:
            print("没有有效的k-space文件")
            return False

        print(f"准备处理 {len(valid_paths)} 个k-space文件")

        # 使用多进程并行处理
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # 为每个任务添加参数
            tasks = [(file_path, mask, wavelet_params) for file_path in valid_paths]
            results = pool.map(process_wrapper_kspace, tasks)

        # 保存结果
        success_count = 0
        for file_path, result in zip(valid_paths, results):
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

        print(f"处理完成，成功处理 {success_count}/{len(valid_paths)} 个文件")
        return results

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


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


# 主函数
def main():
    print("=== MRI压缩感知重建批量处理程序 ===")

    # 用户选择输入文件夹
    input_folder = select_folder("选择k-space数据文件夹")
    if not input_folder:
        print("未选择输入文件夹，程序退出")
        return

    # 用户选择mask文件
    mask_path = select_file("选择mask文件")
    if not mask_path:
        print("未选择mask文件，程序退出")
        return

    # 用户选择输出文件夹
    output_folder = select_folder("选择输出文件夹")
    if not output_folder:
        print("未选择输出文件夹，程序退出")
        return

    print(f"输入文件夹: {input_folder}")
    print(f"Mask文件: {mask_path}")
    print(f"输出文件夹: {output_folder}")

    # 第一步：运行批量处理
    results = process_kspace_files(input_folder, mask_path, output_folder, save_npy=True)


    # 可视化第一个有效结果作为示例
    valid_results = [result for result in results if result is not None]
    print(f"成功处理 {len(valid_results)} 个文件")

    if valid_results:
        reconstructed = valid_results[0]

        # 显示欠采样图像和重建图像
        plt.figure(figsize=(12, 6))

        plt.imshow(reconstructed, cmap='gray')
        plt.title("Recon Iamge")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'sample_reconstruction.png'))
        print("已保存示例重建图像")

        plt.show()
    else:
        print("没有有效结果可供可视化")


if __name__ == "__main__":
    main()