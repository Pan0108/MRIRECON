# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QFileDialog, QMessageBox)
from PIL import Image
import struct



def readfid(filename: str, channel: int = 0):
    if channel < 0 or channel > 1:
        channel = 0

    file = open(filename, 'rb')
    file_version = int.from_bytes(file.read(4), byteorder='little')
    section1_size = int.from_bytes(file.read(4), byteorder='little')
    section2_size = int.from_bytes(file.read(4), byteorder='little')

    position = section1_size + section2_size + 4 * 3
    file.seek(position, 1)
    dimension1 = int.from_bytes(file.read(4), byteorder='little')
    dimension2 = int.from_bytes(file.read(4), byteorder='little')
    dimension3 = int.from_bytes(file.read(4), byteorder='little')
    dimension4 = int.from_bytes(file.read(4), byteorder='little')

    dataBytes = file.read(2 * 4 * dimension1 * dimension2 * dimension3 * dimension4)
    data = np.array(struct.unpack('f' * 2 * dimension1 * dimension2 * dimension3 * dimension4, dataBytes))

    raw = data[0::2] + 1j * data[1::2];
    raw = np.squeeze(np.reshape(raw, (dimension1, dimension2, dimension3, dimension4), order='F'))

    if channel == 0:
        return raw

    dataBytes = file.read(2 * 4 * dimension1 * dimension2 * dimension3 * dimension4)
    data = np.array(struct.unpack('f' * 2 * dimension1 * dimension2 * dimension3 * dimension4, dataBytes))

    raw2 = data[0::2] + 1j * data[1::2];
    raw2 = np.squeeze(np.reshape(raw, (dimension1, dimension2, dimension3, dimension4), order='F'))
    return raw, raw2

def dim_match(A_shape, B_shape):
    """维度匹配函数"""
    A_out_shape = A_shape
    B_out_shape = B_shape
    if len(A_shape) < len(B_shape):
        for _ in range(len(A_shape), len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape), len(A_shape)):
            B_out_shape += (1,)
    return A_out_shape, B_out_shape


def FFt_forward(im, mask, axes=(0, 1)):
    """FFT前向变换"""
    im = np.fft.fftshift(im, axes)
    ksp = np.fft.fft2(im, s=None, axes=axes)
    ksp = np.fft.ifftshift(ksp, axes)
    if len(ksp.shape) != len(mask.shape):
        ksp_out_shape, mask_out_shape = dim_match(ksp.shape, mask.shape)
        mksp = np.multiply(ksp.reshape(ksp_out_shape), mask.reshape(mask_out_shape))
    else:
        mksp = np.multiply(ksp, mask)
    return mksp


def select_input_file():
    """交互式选择输入文件 (Qt版本)"""
    app = QApplication.instance() or QApplication([])
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "选择FID数据文件",
        "E:\\PLX20250611",
        "FID files (**.fid);;All files (*.*)"
    )
    return file_path


def process_slice(slice_data, output_path, base_name, slice_idx):
    """处理单个切片并保存"""
    x = abs(np.fft.fftshift(np.fft.ifft2((slice_data.T), (256, 256))))
    x = np.roll(x, 128, 0)[::-1, :]  # MCSIII

    # 归一化并转换
    x_scaled = (x - x.min()) / (x.max() - x.min())
    im = (x_scaled * 255).astype(np.uint8)

    # 保存PNG图像
    file_path = os.path.join(output_path, f'{base_name}_slice_{slice_idx}.png')

    Image.fromarray(im).save(file_path)

    return file_path


def execute_transform(input_file):

    # 2. 自动创建输出目录 (在输入文件同级目录下创建Fid2images文件夹)
    input_dir = os.path.dirname(input_file)
    output_folder = os.path.join(input_dir, "Fid2images")
    os.makedirs(output_folder, exist_ok=True)

    # 3. 确认操作
    reply = QMessageBox.question(
        None,
        "确认",
        f"将处理文件:\n{input_file}\n输出到:\n{output_folder}\n是否继续?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes
    )
    if reply != QMessageBox.Yes:
        print("用户取消操作")
        return

    # 4. 处理数据
    try:
        print(f"开始处理文件: {input_file}")
        ksp = readfid(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        print(f"数据形状: {ksp.shape}")
        for i in range(ksp.shape[2]):
            slice_data = ksp[:, :, i]
            saved_path = process_slice(slice_data, output_folder, base_name, i)
            print(f"已保存切片 {i} 到: {saved_path}")

        QMessageBox.information(None, "完成", "所有切片处理完成！")
    except Exception as e:
        QMessageBox.critical(None, "错误", f"处理过程中发生错误:\n{str(e)}")


if __name__ == '__main__':
    # 1. 交互式选择输入文件
    input_file = select_input_file()

    execute_transform(input_file)