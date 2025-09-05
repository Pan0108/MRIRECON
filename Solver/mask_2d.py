# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import pynufft as nf
'''
默认核磁共振k空间数据填充，比如相位方向随机，径向采样，中心半径保留2D随机，spiral轨迹采样等等
'''
def mask2d_phase(image_size=256, radius=15,undersampling=0.5,axis=1):  # 假设相位方向半径为10，其余部分随机选择
    np.random.seed(0)
    # 获取图像尺寸
    shape = (image_size,image_size)
    height, width = shape
    # 确定中心点的列索引
    center_col = int(shape[axis] // 2)
    # 计算固定取值的列范围
    start_col = center_col - radius
    end_col = center_col + radius + 1  # 包含半径外的边界
    # 初始化一个全零的掩码
    ma = np.zeros(shape, dtype=bool) # 使用bool类型更直观

    # 计算剩余需要欠采样的行/列数
    remaining_cols = shape[axis] - (end_col - start_col)
    num_samples = int(remaining_cols * undersampling)
    # 随机选择欠采样的列索引（不包括中心半径范围内的列）
    remaining_indices = list(range(start_col)) + list(range(end_col, shape[axis]))
    sampled_indices = np.random.choice(remaining_indices, num_samples, replace=False)
    # 设置随机选择的列掩码为True
    # 设置中心半径范围内的掩码为True
    if axis == 1:
        ma[:, start_col:end_col] = True
        ma[:, sampled_indices] = True
    else:
        ma[start_col:end_col, :] = True
        ma[sampled_indices,:] = True

    return ma


def mask2d_normal(shape, radius, interval=2, axis=1):
    """
    生成一个掩码矩阵，中心区域和按固定间隔采样的列为 True。

    参数：
        shape (tuple): 图像的形状 (height, width)。
        radius (int): 中心区域的半径。
        interval (int): 固定间隔大小（每隔 interval 列采样一次）。
        axis (int): 指定操作的轴（1 表示列，0 表示行）。

    返回：
        ma (ndarray): 布尔类型的掩码矩阵。
    """
    # 初始化一个全零的掩码
    ma = np.zeros(shape, dtype=bool)

    # 获取图像尺寸
    height, width = shape

    # 确定中心点的列索引
    center_col = int(shape[axis] // 2)

    # 计算固定取值的列范围（中心区域）
    start_col = center_col - radius
    end_col = center_col + radius + 1  # 包含半径外的边界

    # 设置中心半径范围内的掩码为 True
    if axis == 1:
        ma[:, start_col:end_col] = True
    else:
        ma[start_col:end_col, :] = True

    # 生成按固定间隔采样的列索引
    sampled_indices = list(range(0, shape[axis], interval))  # 每隔 interval 列采样

    # 排除中心区域的列索引
    remaining_indices = [idx for idx in sampled_indices if idx < start_col or idx >= end_col]

    # 设置选中的列掩码为 True
    if axis == 1:
        ma[:, remaining_indices] = True
    else:
        ma[remaining_indices, :] = True

    return ma

def mask2d_center(image_size=256, center_r=15, undersampling=0.3):  #模拟采样数据，对k空间数据进行欠采样
    #创建一个形状一致的mask
    shape = (image_size,image_size)
    nx,ny = shape
    # 降采样点数
    k = int(round(nx*ny*undersampling))
    # 降采样点的索引
    ri = np.random.choice(nx*ny,k,replace=False)
    # 初始化mask
    ma = np.zeros(nx*ny)
    # 将采样点设置为1，或bool--TRUE
    ma[ri] = 1
    mask = ma.reshape((nx,ny))

    # kspace的中心
    if center_r > 0:
        cx = int(nx/2)
        cy = int(ny/2)
        cxr_min, cxr_max = max(0, cx - center_r), min(nx, cx + center_r + 1)
        cyr_min, cyr_max = max(0, cy - center_r), min(ny, cy + center_r + 1)
        # 然后，设置中心区域的值
        mask[cxr_min:cxr_max, cyr_min:cyr_max] = 1
    return mask


def mask2d_radial(image_size=256, num_rays=None, undersampling=None):
    """
    生成2D k空间的径向采样掩码。

    参数:
    shape (tuple): k空间的形状，例如 (256, 256)。
    num_rays (int): 径向线的数量。
    undersampling (float): 欠采样率，控制每条射线上的采样间距。

    返回:
    np.ndarray: 径向采样掩码，形状与k空间相同，True表示采样点。
    """
    shape = (image_size,image_size)
    height, width = shape
    center_row, center_col = height // 2, width // 2
    mask = np.zeros(shape, dtype=bool)

    # 计算从中心到边缘的最大距离
    # max_radius = np.sqrt((height // 2) ** 2 + (width // 2) ** 2)
    max_radius = min(height,width)/2
    # 计算每条射线的总步长（步数 = 最大半径 * 欠采样率）
    num_steps = int(max_radius * undersampling)

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    for angle in angles:
        # 单位方向向量
        dir_x = np.cos(angle)
        dir_y = np.sin(angle)

        # 沿正负两个方向采样
        for direction in [1, -1]:
            x, y = center_row, center_col  # 起始于中心点
            for _ in range(num_steps):
                # 计算当前坐标并四舍五入
                xi = int(round(x))
                yi = int(round(y))
                if 0 <= xi < height and 0 <= yi < width:
                    mask[xi, yi] = True
                # 沿着方向移动一个步长（步长=1/undersampling，方向由direction控制）
                x += dir_x * direction
                y += dir_y * direction

    return mask



def traj2d_radial(image_size=256,num_rays=None,undersampling=None):
    """改进：确保轨迹点连续覆盖全范围"""
    shape = (image_size,image_size)
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




def mask2d_spiral(image_size=256, num_turns=None, undersampling=1, spacing_power=None):
    """
    生成从内到外间距逐渐增大的螺旋采样掩码
    参数:
    shape (tuple): k空间的形状，例如 (256, 256)
    num_turns (int): 螺旋的总圈数
    undersampling (float): 欠采样率（控制总采样点数）,更应该说是密度较为合适
    spacing_power (float): 间距增长指数（>1时外圈间距更大，建议2-3）
    返回:
    np.ndarray: 螺旋采样掩码，True表示采样点
    """
    shape = (image_size,image_size)
    height, width = shape
    center_row, center_col = height // 2, width // 2
    mask = np.zeros(shape, dtype=bool)
    max_radius = min(height, width) / 2

    # 计算总步数
    total_steps = int(max_radius * undersampling * num_turns * 1.5)  # 适当增加总步数

    # 生成非线性增长的半径
    t = np.linspace(0, 1, total_steps)
    radii = max_radius * (t ** spacing_power)  # 关键修改：半径按幂函数增长

    # 生成角度（保持总圈数不变）
    angles = np.linspace(0, 2 * np.pi * num_turns, total_steps)

    for radius, angle in zip(radii, angles):
        x = center_row + radius * np.cos(angle)
        y = center_col + radius * np.sin(angle)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < height and 0 <= yi < width:
            mask[xi, yi] = True

    return mask



def traj2d_spiral(image_size=256, num_turns=None, undersampling=1, spacing_power=None):
    """
    生成从内到外间距逐渐增大的螺旋采样轨迹（坐标形式）

    参数:
        shape (tuple): k空间的形状，例如 (256, 256)
        num_turns (int): 螺旋的总圈数
        undersampling (float): 控制采样点密度（值越小，点越稀疏）
        spacing_power (float): 间距增长指数（>1时外圈间距更大，建议2-3）

    返回:
        np.ndarray: 螺旋轨迹坐标，形状为 (N, 2)，范围归一化到 [-0.5, 0.5]
    """
    shape = (image_size,image_size)
    height, width = shape
    center_row, center_col = height / 2, width / 2
    max_radius = min(height, width) / 2

    # 计算总步数（根据欠采样率和圈数调整）
    total_steps = int(max_radius * undersampling * num_turns * 1.5)

    # 生成非线性增长的半径和角度
    t = np.linspace(0, 1, total_steps)
    radii = max_radius * (t ** spacing_power)  # 半径按幂函数增长
    angles = np.linspace(0, 2 * np.pi * num_turns, total_steps)

    # 生成轨迹坐标
    coords = []
    for radius, angle in zip(radii, angles):
        x = center_col + radius * np.cos(angle)  # 注意：x对应width（列）
        y = center_row + radius * np.sin(angle)  # y对应height（行）
        if 0 <= x < width and 0 <= y < height:
            coords.append((x - center_col, y - center_row))  # 中心化坐标

    # 归一化到[-0.5, 0.5]
    kspace_coords = np.array(coords, dtype=np.float32)
    kspace_coords[:, 0] = kspace_coords[:, 0] * (1 / width)
    kspace_coords[:, 1] = kspace_coords[:, 1] * (1 / height)

    return kspace_coords


def sample_kdata(full_kspace, coords):
    """从全采样k空间中提取非笛卡尔采样点 (并行加速版)"""
    height, width = full_kspace.shape
    kdata = np.zeros(len(coords), dtype=np.complex64)

    def interpolate_point(i):
        kx, ky = coords[i]
        x0, y0 = int(np.floor(kx)), int(np.floor(ky))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        # 边界处理
        x0 = max(0, x0)
        y0 = max(0, y0)

        # 双线性插值
        dx, dy = kx - x0, ky - y0
        return (full_kspace[y0, x0] * (1 - dx) * (1 - dy) +
                full_kspace[y0, x1] * dx * (1 - dy) +
                full_kspace[y1, x0] * (1 - dx) * dy +
                full_kspace[y1, x1] * dx * dy)

    # 使用线程池并行计算
    with concurrent.futures.ThreadPoolExecutor() as executor:
        kdata = list(executor.map(interpolate_point, range(len(coords))))
    print(f"kdata的形状{np.array(kdata).shape},coords的形状{coords.shape}")
    return np.array(kdata)


def kspace_nufft(image,traj,shape=(256,256)):
    # 初始化NUFFT对象
    nufft_obj = nf.NUFFT()
    Kd = (2 * shape[0], 2 * shape[1])  # 过采样size
    nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))

    # 生成k空间数据
    kspace = nufft_obj.forward(image.astype(np.complex64))
    print(f"kspace的形状{np.array(kspace).shape},coords的形状{traj.shape}")
    return kspace


if __name__ == '__main__':
    shape = (256,256)

    # mask = mask2d_phase(shape, undersampling=0.2,radius=15,axis=1) # 1D random
    # mask = mask2d_radial(shape, num_rays=64, undersampling=1)  # radial random
    # mask = mask2d_center(shape, center_r=10, undersampling=0.5) # 2D random
    # mask = mask2d_spiral(shape, num_turns=20, undersampling=1, spacing_power=2)
    # mask = mask2d_normal(image_size=256, radius=15, interval=3, axis=1) #1D normal
    # traj = traj2d_spiral(image_size=256, num_turns=10, undersampling=0.75, spacing_power=2)
    # 保存掩码为.npy文件
    # np.save('mask2d_phase.npy', mask)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Radial Sampling Mask')
    # plt.scatter(traj[:, 0], traj[:, 1], s=1, c="b",marker=1)  # 正确写法
    # plt.title("Spiral Trajectory (Normalized)")
    # plt.xlabel("kx")
    # plt.ylabel("ky")
    #
    # plt.grid(True)
    # plt.gca().set_aspect('equal')  # 关键修正：强制 1:1 比例
    # plt.show()
    from PIL import Image
    img_path = r"C:\Users\lixing.pan\Desktop\data_1\val_nufft\full\chengzi0328_1_slice_0_nturn50.png"
    img = Image.open(img_path).convert('L').resize((256,256))  # 转灰度图并调整大小
    image = np.array(img) / 255.0  # 归一化到 [0, 1]

    traj = r"C:\Users\lixing.pan\Desktop\data_1\val_nufft\mask\chengzi0328_1_slice_0_nturn50_mask.npy"
    traj = np.load(traj)
    kspace = kspace_nufft(image,traj)
    # 初始化NUFFT对象
    nufft_obj = nf.NUFFT()
    Kd = (2 * shape[0], 2 * shape[1])  # 过采样size
    nufft_obj.plan(traj, shape, Kd, Jd=(6, 6))
    recon_image = np.abs(nufft_obj.adjoint(kspace))

    plt.imshow(recon_image, cmap='gray')
    plt.title('Radial Sampling Mask')
    plt.show()

    # np.save(r"C:\Users\lixing.pan\Desktop\data_1\val_nufft\undersampled\chengzi0328_1_slice_0_nturn50_kspace.npy",kspace)