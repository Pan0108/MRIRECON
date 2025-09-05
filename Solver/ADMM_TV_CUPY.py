# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import os
from PIL import Image

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

def FFt_forward(im, mask, axes=(0,1)):
    ksp = cp.fft.fft2(im, s=None, axes=axes)
    if len(ksp.shape) != len(mask.shape):
        ksp_out_shape, mask_out_shape = dim_match(ksp.shape, mask.shape)
        mksp = cp.multiply(ksp.reshape(ksp_out_shape), mask.reshape(mask_out_shape))
    else:
        mksp = cp.multiply(ksp, mask)
    return mksp

def FFt_backward(ksp, axes=(0,1)):
    im = cp.fft.ifft2(ksp, s=None, axes=axes)
    return im


class TV2d_r:
    "this defines functions related to total variation minimization using CuPy"
    def __init__(self):
        self.ndim = 2  # number of image dimensions

    def grad(self, x):
        "Compute the gradient of x using CuPy"
        sx = x.shape[0]
        sy = x.shape[1]
        # Generate indices with CuPy
        rows = cp.concatenate([cp.arange(1, sx), cp.array([sx-1])])
        Dx = x[rows, :] - x
        self.rx = x[sx-1, :].copy()  # 存储边界值
        # Handle columns
        cols = cp.concatenate([cp.arange(1, sy), cp.array([sy-1])])
        Dy = x[:, cols] - x
        self.ry = x[:, sy-1].copy()
        # Create result array
        res = cp.zeros(x.shape + (self.ndim,), dtype=x.dtype)
        res[..., 0] = Dx
        res[..., 1] = Dy
        return res

    def adjgradx(self, x):
        "Adjoint gradient operation for x-axis"
        sx = x.shape[0]
        x = x.copy()  # 避免修改原数组
        x[sx-1, :] = self.rx  # 恢复边界值
        x = cp.flip(cp.cumsum(cp.flip(x, 0), 0))
        return x

    def adjgrady(self, x):
        "Adjoint gradient operation for y-axis"
        sy = x.shape[1]
        x = x.copy()
        x[:, sy-1] = self.ry
        x = cp.flip(cp.cumsum(cp.flip(x, 1), 1), 1)
        return x

    def adjgrad(self, y):
        "Adjoint gradient combined operation"
        return self.adjgradx(y[..., 0]) + self.adjgrady(y[..., 1])

    def adjDy(self, x):
        "Adjoint Dy operator for divergence calculation"
        sy = x.shape[1]
        cols = cp.concatenate([cp.array([0]), cp.arange(0, sy-1)])
        res = x[:, cols] - x
        res[:, 0] = -x[:, 0]
        res[:, -1] = x[:, -2]
        return res

    def adjDx(self, x):
        "Adjoint Dx operator for divergence calculation"
        sx = x.shape[0]
        rows = cp.concatenate([cp.array([0]), cp.arange(0, sx-1)])
        res = x[rows, :] - x
        res[0, :] = -x[0, :]
        res[-1, :] = x[-2, :]
        return res

    def Div(self, y):
        "Compute divergence using adjoint operators"
        return self.adjDx(y[..., 0]) + self.adjDy(y[..., 1])

    def amp(self, grad):
        "Compute amplitude of gradient vectors"
        amp = cp.sqrt(cp.sum(grad**2, axis=-1))  # 沿最后一个轴求和
        amp_shape = amp.shape + (1,)
        d = cp.ones(amp_shape, dtype=amp.dtype) * amp.reshape(amp_shape)
        return d

    # Image -> sparse domain
    def backward(self, x):
        return self.grad(x)

    # Sparse domain -> image
    def forward(self, y):
        return self.Div(y)


def prox_tv2d_r(y, lambda_tv, step=0.1):
    sizeg = y.shape + (2,)
    G = cp.zeros(sizeg)
    i = 0
    tvopt = TV2d_r()
    while i < 40:
        dG = tvopt.grad(tvopt.Div(G) - y / lambda_tv)
        G = G - step * dG
        d = tvopt.amp(G)
        G = G / cp.maximum(d, cp.ones(sizeg))
        i += 1
    f = y - lambda_tv * tvopt.Div(G)
    return f

def prox_l2_Afxnb_CGD(mask, Afunc, invAfunc, b, x0, rho, Nite, ls_Nite=10):
    eps = 0.001
    i = 0

    def f(xi):
        return cp.linalg.norm(Afunc(xi, mask) - b)**2 + (rho/2)*cp.linalg.norm(xi - x0)**2

    def df(xi, mask):
        return 2*invAfunc(Afunc(xi, mask) - b) + rho*(xi - x0)

    dx = -df(x0, mask)
    alpha, nstp = BacktrackingLineSearch(mask, f, df, x0, dx, ls_Nite=ls_Nite)
    x = x0 + alpha * dx
    s = dx
    delta0 = cp.linalg.norm(dx)
    deltanew = delta0

    while i < Nite and deltanew > eps*delta0 and nstp < ls_Nite:
        dx = -df(x, mask)
        deltaold = deltanew
        deltanew = cp.linalg.norm(dx)
        beta = float(deltanew / deltaold)
        s = dx + beta * s
        alpha, nstp = BacktrackingLineSearch(mask, f, df, x, s, ls_Nite=ls_Nite)
        x = x + alpha * s
        i += 1
    return x

def BacktrackingLineSearch(mask, f, df, x, p, c=0.0001, rho=0.2, ls_Nite=10):
    derphi = cp.real(cp.dot(p.flatten(), cp.conj(df(x, mask)).flatten())).get()
    f0 = f(x).get()
    alphak = 1.0
    f_try = f(x + alphak * p).get()
    i = 0

    while i < ls_Nite and (f_try - f0) > c * alphak * derphi and f_try > f0:
        alphak *= rho
        f_try = f(x + alphak * p).get()
        i += 1
    return cp.float32(alphak), i

def run_test(b, mask, Afunc=FFt_forward, invAfunc=FFt_backward, tv_r=10, rho=1, n_iter=50, step=0.5, cg_iter=3, tv_ndim=2):
    z = FFt_backward(b)
    u = cp.zeros_like(z)
    tvprox = prox_tv2d_r if tv_ndim == 2 else None
    gr_ADMM = []

    for _ in range(n_iter):
        x = prox_l2_Afxnb_CGD(mask, Afunc, invAfunc, b, z - u, rho, cg_iter)
        z = tvprox(x + u, 2.0 * tv_r / rho)
        u = u + step * (x - z)
        gr_ADMM.append(cp.linalg.norm(x - z).get())
        print(f'gradient in ADMM {gr_ADMM[-1]}')
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

def process_single_kspace(file_path, mask, TV_Cupy_params=None):
    """
    处理单个k-space文件

    参数:
        file_path: k-space文件路径
        mask: 掩码数据
        wavelet_params: 小波参数字典，包含以下键:
         tv_r=5,  # TV正则化参数
             rho=1,  # ADMM惩罚参数
             n_iter=30,  # 迭代次数
             step=0.5,  # 步长
             cg_iter=3,  # 共轭梯度迭代次数
             tv_ndim=2  # 2D TV
    """
    # 设置默认参数
    default_params = {
        "tv_r": 5,
        "rho": 1,
        "n_iter": 50,
        "step": 0.5,
        "cg_iter": 3,
        "tv_ndim": 2,
    }

    # 合并默认参数和传入参数
    if TV_Cupy_params is None:
        TV_Cupy_params = default_params
    else:
        TV_Cupy_params = {**default_params, **TV_Cupy_params}

    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        reconstructed = run_test(
            kspace, mask, Afunc=FFt_forward, invAfunc=FFt_backward,
            tv_r=TV_Cupy_params['tv_r'],
            rho=TV_Cupy_params['rho'],
            n_iter=TV_Cupy_params['n_iter'],
            step=TV_Cupy_params['step'],
            cg_iter=TV_Cupy_params['cg_iter'],
            tv_ndim=TV_Cupy_params['tv_ndim']
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
    file_path, mask, TV_Cupy_params = args
    try:
        return process_single_kspace(file_path, mask, TV_Cupy_params)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def process_kspace_files(kspace_paths, mask_path, output_folder, save_npy=True, TV_Cupy_params=None):
    """
    处理多个k-space文件并保存结果

    参数:
        kspace_paths: k-space .npy文件路径列表
        mask_path: mask文件的路径
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
        TV_Cupy_params: TV_Cupy参数字典
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

                # 直接调用处理函数
                result = process_wrapper_kspace((file_path, mask, TV_Cupy_params))
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



if __name__ == '__main__':

    from matplotlib.pyplot import plt
    # 加载数据并转换为CuPy数组
    kspace = cp.asarray(np.load('slice_000_0.5_kspace.npy')[0, :, :])
    mask = cp.asarray(np.load('slice_000_0.5_mask.npy'))
    print(max(cp.abs(kspace.flatten())))

    b = kspace*255
    im = cp.abs(cp.fft.ifft2(b))
    x0 = FFt_backward(b)

    x1, gr_ADMM = run_test(b, mask, FFt_forward, FFt_backward, Nite=30, step=0.5, tv_r=10, rho=1, cgd_Nite=3, tvndim=2)

    # 转换为NumPy数组用于显示
    mask_np = mask.get()
    im_np = im.get()
    x0_np = x0.get()
    x1_np = x1.get()

    plt.figure(figsize=(12, 6))
    plt.imshow(cp.asnumpy(cp.abs(mask)), cmap='gray')
    plt.title('k-space mask')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(gr_ADMM)
    plt.title('Gradient in ADMM')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cp.asnumpy(cp.abs(x0)), cmap='gray')
    plt.title("Undersampled Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cp.asnumpy(cp.abs(x1)), cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')

    plt.show()