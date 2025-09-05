# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os

class TV2d_r:
    def __init__( self ):
        self.ndim    = 2
    def grad(self, x):
        sx = x.shape[0]
        sy = x.shape[1]
        Dx = x[np.r_[1:sx, sx-1],:] - x
        self.rx = x[sx-1,:]
        Dy = x[:,np.r_[1:sy, sy-1]] - x
        self.ry = x[:,sy-1]
        res = np.zeros(x.shape + (self.ndim,), dtype = x.dtype)
        res[...,0] = Dx
        res[...,1] = Dy
        return res

    def adjgradx(self, x ):
        sx   = x.shape[0]
        x[sx-1,:] = self.rx
        x = np.flip(np.cumsum(np.flip(x,0), 0),0)

        return x

    def adjgrady( self, x ):
        sy = x.shape[1]
        x[:,sy-1] = self.ry
        x = np.flip(np.cumsum(np.flip(x,1), 1),1)

        return x

    def adjgrad( self, y ):
        res = self.adjgradx(y[...,0]) + self.adjgrady(y[...,1])
        return res

    def adjDy( self, x ):
        sx = x.shape[0]
        sy = x.shape[1]
        res = x[:,np.r_[0, 0:sy-1]] - x
        res[:,0] = -x[:,0]
        res[:,-1] = x[:,-2]
        return res

    def adjDx( self, x ):
        sx = x.shape[0]
        sy = x.shape[1]
        res = x[np.r_[0, 0:sx-1],:] - x
        res[0,:] = -x[0,:]
        res[-1,:] = x[-2,:]
        return res

    def Div( self, y ):
        res = self.adjDx(y[...,0]) + self.adjDy(y[...,1])
        return res

    def amp( self, grad ):
        amp = np.sqrt(np.sum(grad ** 2, axis=(len(grad.shape)-1)))
        amp_shape = amp.shape + (1,)

        d = np.ones(amp.shape + (self.ndim,), dtype = amp.dtype)
        d = np.multiply(amp.reshape(amp_shape), d)
        return d
    # image --> sparse domain
    def backward( self, x ):
        return self.grad(x)

    def forward( self, y ):
        return self.Div(y)


class MRIReconstructor:
    def __init__(self, tv_r=5, rho=1, n_iter=30, step=0.5, cg_iter=3, tv_ndim=2):
        """
        初始化MRI重建器

        参数:
            tv_r: TV正则化参数
            rho: ADMM惩罚参数
            n_iter: ADMM迭代次数
            step: ADMM步长
            cg_iter: 共轭梯度迭代次数
            tv_ndim: TV维度 (2或3)
        """
        self.params = {
            'tv_r': tv_r,
            'rho': rho,
            'n_iter': n_iter,
            'step': step,
            'cg_iter': cg_iter,
            'tv_ndim': tv_ndim
        }

    @staticmethod
    def dim_match(A_shape, B_shape):
        """确保两个数组形状匹配，通过添加长度为1的维度"""
        A_out, B_out = A_shape, B_shape
        if len(A_shape) < len(B_shape):
            A_out += (1,) * (len(B_shape) - len(A_shape))
        elif len(A_shape) > len(B_shape):
            B_out += (1,) * (len(A_shape) - len(B_shape))
        return A_out, B_out

    def FFt_forward(self, im, mask, axes=(0, 1)):
        """正向FFT变换并应用采样掩码"""
        im = np.fft.fftshift(im, axes)
        ksp = np.fft.fft2(im, s=None, axes=axes)
        ksp = np.fft.ifftshift(ksp, axes)

        if len(ksp.shape) != len(mask.shape):
            ksp_shape, mask_shape = self.dim_match(ksp.shape, mask.shape)
            mksp = ksp.reshape(ksp_shape) * mask.reshape(mask_shape)
        else:
            mksp = ksp * mask
        return mksp

    def FFt_backward(self, ksp, axes=(0, 1)):
        """逆向FFT变换"""
        ksp = np.fft.fftshift(ksp, axes)
        im = np.fft.ifft2(ksp, s=None, axes=axes)
        im = np.fft.ifftshift(im, axes)
        return im

    def prox_tv2d_r(self, y, lambda_tv, step=0.1):
        """2D TV正则化近端算子"""
        sizeg = y.shape + (2,)
        G = np.zeros(sizeg)
        tvopt = TV2d_r()

        for _ in range(40):
            dG = tvopt.grad(tvopt.Div(G) - y / lambda_tv)
            G = G - step * dG
            d = tvopt.amp(G)
            G /= np.maximum(d, 1.0)

        return y - lambda_tv * tvopt.Div(G)


    def prox_l2(self, Afunc, invAfunc, b, x0, rho, Nite, mask, ls_Nite=10):
        """L2近端算子(使用共轭梯度法)"""

        def f(xi):
            return np.linalg.norm(Afunc(xi, mask) - b) ** 2 + (rho / 2) * np.linalg.norm(xi - x0) ** 2

        def df(xi):
            return 2 * invAfunc(Afunc(xi, mask) - b) + rho * (xi - x0)

        # 共轭梯度优化
        dx = -df(x0)
        alpha, nstp = self.backtracking_line_search(f, df, x0, dx, ls_Nite)
        x = x0 + alpha * dx
        s = dx
        delta0 = np.linalg.norm(dx)

        for _ in range(Nite):
            dx = -df(x)
            deltanew = np.linalg.norm(dx)
            beta = deltanew / delta0
            s = dx + beta * s
            alpha, nstp = self.backtracking_line_search(f, df, x, s, ls_Nite)
            x += alpha * s
            delta0 = deltanew

        return x

    def backtracking_line_search(self, f, df, x, p, max_iter=10, c=0.0001, rho=0.2):
        """回溯线搜索"""
        derphi = np.real(np.dot(p.flatten(), np.conj(df(x)).flatten()))
        f0 = f(x)
        alpha = 1.0

        for _ in range(max_iter):
            f_try = f(x + alpha * p)
            if f_try <= f0 + c * alpha * derphi:
                break
            alpha *= rho

        return alpha, _


    def reconstruct(self, kspace, mask):
        """
        执行MRI重建

        参数:
            kspace: 降采样的k空间数据
            mask: 采样掩码

        返回:
            重建后的图像
            收敛曲线
        """
        # 初始化
        z = self.FFt_backward(kspace)
        u = np.zeros_like(z)

        # 选择TV近端算子
        if self.params['tv_ndim'] == 2:
            tv_prox = self.prox_tv2d_r
        else:
            raise ValueError("不支持的TV维度")

        # ADMM迭代
        for _ in range(self.params['n_iter']):
            x = self.prox_l2(
                self.FFt_forward, self.FFt_backward,
                kspace, z - u, self.params['rho'],
                self.params['cg_iter'], mask
            )
            z = tv_prox(x + u, 2.0 * self.params['tv_r'] / self.params['rho'])
            u += self.params['step'] * (x - z)

            print('gradient in ADMM %g' % np.linalg.norm(x-z))

        return np.abs(x)



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

def process_single_kspace(file_path, mask, TV_params=None):
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
    if TV_params is None:
        TV_params = default_params
    else:
        TV_params = {**default_params, **TV_params}

    try:
        print(f"处理文件: {file_path}")
        # 加载k-space数据
        kspace = np.load(file_path)

        # 运行重建算法
        MRIRecon = MRIReconstructor(
            tv_r=TV_params['tv_r'],
            rho=TV_params['rho'],
            n_iter=TV_params['n_iter'],
            step=TV_params['step'],
            cg_iter=TV_params['cg_iter'],
            tv_ndim=TV_params['tv_ndim']
        )
        reconstructed = MRIRecon.reconstruct(
            kspace, mask)


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
    file_path, mask, TV_params = args
    try:
        return process_single_kspace(file_path, mask, TV_params)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def process_kspace_files(kspace_paths, mask_path, output_folder, save_npy=True, TV_params=None):
    """
    处理多个k-space文件并保存结果

    参数:
        kspace_paths: k-space .npy文件路径列表
        mask_path: mask文件的路径
        output_folder: 输出文件夹路径
        save_npy: 是否保存npy格式文件
        TV_params: TV参数
    """

    try:
        # 检查/创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 加载mask
        mask = np.load(mask_path)
        print(f"加载mask文件: {mask_path}, 形状: {mask.shape}")

        print(f"准备处理 {len(kspace_paths)} 个k-space文件")

        if not kspace_paths:
            print("没有提供k-space文件路径")
            return False

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
                result = process_wrapper_kspace((file_path, mask, TV_params))
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
    """
    主程序调用上面的模块： 1、创建重建器， reconstructor = MRIReconstructor(。。。。。）
                       2、执行重建， x1 = reconstructor.reconstruct(kspace,mask)  #x1为重建图像的二维数据格式，传入参数分别为降采样kspace数据 
                          及采样掩蔽轨迹mask;
    
    如果在外面的程序调用本模块中的重建方法：
                 1、导入模块         from ADMM_TV_Module import MRIReconstructor;
                 2、调用执行重建，    x1 = MRIReconstructor（。。。。）.reconstruct(kspace,mask)
                    或这样写         reconstructor = MRIReconstructor(。。。。。）
                                    x1 = reconstructor.reconstruct(kspace,mask)；
                               
    参数说明： mask的要求:一般mask会强制低频采样，也就是中心区域强制采样8%-15%， 对于256的相位编码，根据实际样品形状取20-40条相位编码，也就是radius=10-20；
             根据实际情况，调节低频（框架）与高频（细节）的比率。
             另外图像分辨率会影响重建速度，就本台普通性能笔记本电脑，比如256*256的图像，迭代30次，需要时间16s， 而256*192图像，重建只需要10s
             
             MRIReconstructor（
             tv_r=5,  # TV正则化参数
             rho=1,  # ADMM惩罚参数
             n_iter=30,  # 迭代次数
             step=0.5,  # 步长
             cg_iter=3,  # 共轭梯度迭代次数
             tv_ndim=2  # 2D TV
             ）重建器的参数问题：
             
            我们主要调节n_iter，默认30，对于1D 相位随机mask是可以的。如果是变密度（越往高频采样密度越小）采样如 radial、spiral采样轨迹，那要调高n_iter=50;
            # TV正则化参数, 默认5， 根据图像信噪比与稀疏性不同，可能会用到10，建议开放调参；
            
    算法性能说明：
          对于1D random 的mask， 理论上可以做到加速3-4倍，上述的算法基本能达到理论水平。
          对于 spiral、radial采样轨迹，理论上可以加速5-6倍；
          更高的加速，要考虑结合并行成像、深度学习
    """
    input_folder = r"D:\潘立星\python_work\pythonProject2\CS_Python\algorithm_testing\data_1\underkspace"
    mask_path = r"D:\潘立星\python_work\pythonProject2\CS_Python\algorithm_testing\data_1\mask\slice_000_0.5_mask.npy"
    output_folder = r"C:\Users\lixing.pan\Desktop\data_1"

    process_kspace_files(input_folder, mask_path, output_folder, save_npy=True, TV_params=None)