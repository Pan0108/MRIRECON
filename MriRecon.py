# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QMessageBox,QInputDialog
from Ui_ImageRecon2 import Ui_ImageRecon
from Solver import Wavelet_ISTA,ADMM_TV,NUFFtWavelet,ADMM_TV_CUPY,FFtTransform,NUFFtTransform,Fid2Img,ImgSplit,mask_2d
from pathlib import Path
import os
import shutil
from PIL import Image
import torch
import torch.fft as fft
import numpy as np
import random
from DLCSMRI import Transformer, TransformerInference, MoDLInference, ISTAInference, MoDL, UNetInference, \
    ista,unet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse


class ImageReconApp(QMainWindow):
    def __init__(self):
        super(ImageReconApp,self).__init__()
        self.ui = Ui_ImageRecon()
        self.ui.setupUi(self)

        # 压缩感知-初始化状态变量
        self.index = 0
        self.input_files_cs = ""
        self.mask_path = ""
        self.output_folder = ""

        # 传统重建-初始化状态变量
        self.index_1 = 0
        self.FFT_input_files = ""
        self.traj_path = ""
        self.output_folder_1 = ""

        # 深度学习-初始化状态变量
        self.index_2 = 0
        self.input_folder_DL = ""
        self.output_folder_DL = ""
        self.config_path = "./DL_params.json"
        self.checkpoint_path_UNet = "./model/U_Net.pth"
        self.checkpoint_path_MoDL = "./model/MoDL.pth"
        self.checkpoint_path_Transformer = "./model/Transformer.pth"
        self.checkpoint_path_ISTA = "./model/ISTA.pth"
        self.dataset = ""
        self.dataloader = ""
        self.max_samples = 3



        #数据处理
        self.input_fid_files = ""
        self.input_img_folder = ""
        self.output_img_dir = ""
        self.num_parts = 12
        self.index_combo_4 = 0

        self.input_name_folder = ""
        self.output_name_folder = ""
        self.suffix = ""

        self.input_img_dir = ""
        self.output_kspace_dir = ""
        self.output_mask_dir = ""
        self.image_size = 256

        self.input_base_path = ""
        self.output_base_path = ""

        self.model_name = None
        self.index_3 = 0
        self.validation_ratio = None


        # 连接信号与槽
        self.connect_slots()

    def connect_slots(self):
        """集中管理所有信号槽连接"""
        # 槽函数连接GUI控键  压缩感知界面
        self.ui.pushButton_8.clicked.connect(self.select_input_files)  # 选择文件夹
        self.ui.pushButton_9.clicked.connect(self.select_mask_file)  # 选择文件夹中文件
        self.ui.pushButton_11.clicked.connect(self.execute_reconstruction)  # 执行重建
        self.ui.comboBox_2.currentIndexChanged.connect(self.set_algorithm_type)  #选择ADMM+TV算法或Wavelet+ISTA算法
        # 槽函数连接GUI控键  传统重建界面
        self.ui.pushButton_5.clicked.connect(self.select_input_files_FFT)  # 选择文件夹
        self.ui.pushButton_10.clicked.connect(self.select_traj_file_1)  # 选择文件夹中文件
        self.ui.pushButton_6.clicked.connect(self.execute_reconstruction_1)  # 执行重建
        self.ui.comboBox.currentIndexChanged.connect(self.set_algorithm_type_1)  # 选择FFT变换或者NUFFT变换

        #深度学习
        self.ui.pushButton_13.clicked.connect(self.select_input_folder_DL)  # 选择kspace文件，可多选
        self.ui.pushButton_14.clicked.connect(self.select_output_folder_DL)  # 选择输出文件夹
        self.ui.comboBox_3.currentIndexChanged.connect(self.set_inference_model) #执行推理
        self.ui.pushButton_15.clicked.connect(self.run_inference_and_display)  # 执行深度学习重建

        #数据处理
        self.ui.pushButton_30.clicked.connect(self.create_dataset_structure)  # 选择文件夹创建数据集
        self.ui.pushButton_7.clicked.connect(self.select_fid_files)  # 选择文件夹创建数据集
        self.ui.pushButton_12.clicked.connect(self.execute_transform)  # fid文件转化为图片格式
        self.ui.pushButton_24.clicked.connect(self.select_img_folder)   #加载待分组目录
        self.ui.pushButton_25.clicked.connect(self.execute_split)       #执行分组
        self.ui.pushButton_17.clicked.connect(self.visual)   #可视化mask/traj
        self.ui.pushButton_18.clicked.connect(self.visual)  # 可视化mask/traj
        self.ui.pushButton_19.clicked.connect(self.visual)  # 可视化mask/traj
        self.ui.pushButton_20.clicked.connect(self.visual)  # 可视化mask/traj
        self.ui.pushButton_28.clicked.connect(self.visual)  # 可视化mask/traj
        self.ui.pushButton_29.clicked.connect(self.visual)  # 可视化mask/traj
        self.ui.comboBox_4.currentIndexChanged.connect(self.set_sampling_type)  #选择降采样轨迹，掩码

        self.ui.pushButton_16.clicked.connect(self.select_name_folder)  # 加载文件路径，输出文件路径
        self.ui.pushButton_21.clicked.connect(self.get_suffix_and_rename)  # 执行文件名修改

        self.ui.pushButton_22.clicked.connect(self.set_img_folder) #选择输入文件夹、输出文件夹
        self.ui.pushButton_23.clicked.connect(self.generate_masks)  #生成underkspace与mask/traj

        self.ui.pushButton_27.clicked.connect(self.execute_transfer)  #划分训练集与验证集

        self.ui.comboBox_5.currentIndexChanged.connect(self.set_train_model)  # 选择相应的模型训练
        self.ui.pushButton_26.clicked.connect(self.trainer)  # 训练


##**压缩感知      两种算法：Wavelet-ISTA / ADMM-TV     支持批量导入kspace/mask 的 2D压缩感知，暂不支持基于GPU的cupy重建算法**##
#======================================================================================================================#
    def execute_reconstruction(self):
        """执行重建"""
        if not self.input_files_cs or not self.mask_path:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择输入文件夹和mask文件！")
            return

        # 获取输出文件夹
        output_folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not output_folder:
            return

        self.output_folder = output_folder

        try:
            # 获取保存选项
            save_npy = self.ui.checkBox.isChecked()
            if self.index == 0:
                # 执行TV重建
                QtWidgets.QMessageBox.information(self, "信息", "TV算法执行中...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                # 这里调用TV算法
                results = ADMM_TV.process_kspace_files(
                    self.input_files_cs,
                    self.mask_path,
                    self.output_folder,
                    save_npy=save_npy,
                    TV_params=self.get_TV_params()
                )
            elif self.index == 1:
                # 执行WaveletISTA重建
                QtWidgets.QMessageBox.information(self, "信息", "WaveletISTA算法执行中...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                results = Wavelet_ISTA.process_kspace_files(
                    self.input_files_cs,
                    self.mask_path,
                    self.output_folder,
                    save_npy=save_npy,
                    wavelet_params=self.get_wavelet_params()
                )

            elif self.index == 2:
                # 执行WaveletISTA重建
                QtWidgets.QMessageBox.information(self, "信息", "WaveletNUFFt算法执行中...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                results = NUFFtWavelet.process_kspace_files(
                    self.input_files_cs,
                    self.mask_path,
                    self.output_folder,
                    save_npy=save_npy,
                    nufft_params=self.get_nufft_params()
                )

            elif self.index == 3:
                # 执行ADMM-TV重建，支持GPU  cupy运算
                QtWidgets.QMessageBox.information(self, "信息", "TV（cupy）算法执行中...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                results = ADMM_TV_CUPY.process_kspace_files(
                    self.input_files_cs,
                    self.mask_path,
                    self.output_folder,
                    save_npy=save_npy,
                    TV_Cupy_params=self.get_TV_Cupy_params()
                )

            if results:
                QtWidgets.QMessageBox.information(self, "完成", "图像重建完成！")

                # 可视化第一个有效结果作为示例
                valid_results = [result for result in results if result is not None]
                print(f"成功处理 {len(valid_results)} 个文件")
                # 显示所有有效结果
                self.show_multiple_results(valid_results)

            else:
                QtWidgets.QMessageBox.warning(self, "警告", "重建过程中出现错误！")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"发生错误: {str(e)}")
        finally:
            # 无论成功失败都恢复光标和界面状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def select_input_files(self):
        """选择多个kspace文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择kspace文件（可多选）",
            "",
            "Numpy Files (*.npy)"
        )

        if files:
            self.input_files_cs = files  # 存储为文件列表

            # 显示选中的文件数量，并用省略号显示部分路径
            if len(files) == 1:
                display_text = f"输入文件 (1个):\n{files[0]}"
            else:
                display_text = f"输入文件 ({len(files)}个):\n{files[0]} 等..."

            elided_text = self.ui.label_6.fontMetrics().elidedText(
                display_text,
                QtCore.Qt.ElideMiddle,
                self.ui.label_6.width() - 10,
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_6.setText(elided_text)
            self.ui.label_6.setToolTip("\n".join(files))  # 悬停时显示所有完整路径


    def select_mask_file(self):
        """选择mask文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择mask文件", "", "Numpy Files (*.npy)")
        if file:
            self.mask_path = file
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_7.fontMetrics().elidedText(
                f"输入文件:\n{file}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_7.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_7.setText(elided_text)
            self.ui.label_7.setToolTip(file)  # 悬停时显示完整路径


    def set_algorithm_type(self, index=0):

            self.index = index  #更新算法类型
            if index == 3:  #检查如果是cupy算法。。。
                try:
                    import cupy as cp
                    cp.array([1, 2, 3])  # 简单测试
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "CuPy警告",
                        f"当前环境不支持CUDA：\n{str(e)}\n"
                        "ADMM-TV-Cupy算法需要NVIDIA GPU和CUDA环境。\n"
                        "请切换到其他算法或配置CUDA环境。"
                    )



    def get_tablewidget_data(self, table,row, column):
        """
        获取表格中指定单元格的数据并自动判断类型返回

        参数:
            table: 表格名称
            row: 行索引(从0开始)
            column: 列索引(从0开始)

        返回:
            int/float/str: 根据内容自动返回相应类型
            None: 如果单元格为空
        """
        item = table.item(row, column)

        if item is None or item.text().strip() == "":
            print(f"({row}, {column}) 单元格为空")
            return None

        text = item.text().strip()

        # 尝试解析为数字
        try:
            # 移除可能存在的千分位分隔符
            text_clean = text.replace(',', '')

            # 尝试转换为浮点数
            num = float(text_clean)

            # 判断是否为整数
            if num.is_integer():
                return int(num)
            return num

        except ValueError:
            # 不是数字，返回原始字符串
            return text

    def get_wavelet_params(self):
        params = {
            "epsilon": self.get_tablewidget_data(self.ui.tableWidget_2,1, 0),
            "n_max": self.get_tablewidget_data(self.ui.tableWidget_2,0, 0),
            "tol": self.get_tablewidget_data(self.ui.tableWidget_2,2, 0),
            "decfac": self.get_tablewidget_data(self.ui.tableWidget_2,3, 0),
            "threshold": self.get_tablewidget_data(self.ui.tableWidget_2,4, 0),
            "wavelet": self.get_tablewidget_data(self.ui.tableWidget_2,5, 0),
            "level": self.get_tablewidget_data(self.ui.tableWidget_2,6, 0),
        }

        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        # 检查 wavelet 是否是有效的小波类型
        valid_wavelets = ['haar', 'db1', 'db2', 'db3', 'db4', 'sym2', 'sym3','sym8', 'coif1']
        if params["wavelet"] not in valid_wavelets:
            raise ValueError(f"无效的小波类型: {params['wavelet']}")

        return params

    def get_TV_params(self):  #ADMM-TV算法的参数
        params = {
            "tv_r": self.get_tablewidget_data(self.ui.tableWidget,1,0),
            "rho": self.get_tablewidget_data(self.ui.tableWidget,2,0),
            "n_iter": self.get_tablewidget_data(self.ui.tableWidget,0,0),
            "step": self.get_tablewidget_data(self.ui.tableWidget,3,0),
            "cg_iter": self.get_tablewidget_data(self.ui.tableWidget,4,0),
            "tv_ndim": self.get_tablewidget_data(self.ui.tableWidget,5,0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_nufft_params(self):   #获取NUFFtWavelet算法的参数
        params = {
            "image_size": self.get_tablewidget_data(self.ui.tableWidget_9,0, 0),
            "eps": self.get_tablewidget_data(self.ui.tableWidget_9,1, 0),
            "tol": self.get_tablewidget_data(self.ui.tableWidget_9,2, 0),
            "L": self.get_tablewidget_data(self.ui.tableWidget_9,3, 0),
            "lamda": self.get_tablewidget_data(self.ui.tableWidget_9,4, 0),
            "max_iter": self.get_tablewidget_data(self.ui.tableWidget_9,5, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params


    def get_TV_Cupy_params(self):  #ADMM-TV-Cupy 算法的参数
        params = {
            "tv_r": self.get_tablewidget_data(self.ui.tableWidget_10,1,0),
            "rho": self.get_tablewidget_data(self.ui.tableWidget_10,2,0),
            "n_iter": self.get_tablewidget_data(self.ui.tableWidget_10,0,0),
            "step": self.get_tablewidget_data(self.ui.tableWidget_10,3,0),
            "cg_iter": self.get_tablewidget_data(self.ui.tableWidget_10,4,0),
            "tv_ndim": self.get_tablewidget_data(self.ui.tableWidget_10,5,0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params


    def show_multiple_results(self, valid_results, max_cols=3):
        num_results = len(valid_results)
        if num_results == 0:
            return

        # 清除之前的图形
        self.ui.fig_0.clear()

        # 计算行数和列数
        cols = min(max_cols, num_results)
        rows = (num_results + cols - 1) // cols

        # 创建子图
        if num_results > 1:
            axes = self.ui.fig_0.subplots(rows, cols)
            axes = axes.ravel()  # 展平axes数组方便迭代
        else:
            axes = [self.ui.fig_0.add_subplot(111)]  # 单个结果的情况

        # 绘制每个结果
        for i, (reconstructed) in enumerate(valid_results):
            ax = axes[i]
            ax.imshow(reconstructed, cmap='gray')
            ax.set_title(f"Recon {i + 1}")
            ax.axis('off')

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # 调整子图间距
        self.ui.fig_0.tight_layout()

        # 重绘画布
        self.ui.canvas_0.draw()



##** 传统重建**##  支持笛卡尔采样重建（FFt变换）与 非笛卡尔采样重建（NUFFt变换）
#========================================================================================================================#

    def execute_reconstruction_1(self):
        """执行重建"""
        if not self.FFT_input_files:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择输入文件")
            return

        # 获取输出文件夹
        output_folder_1 = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not output_folder_1:
            return

        self.output_folder_1 = output_folder_1

        try:
            # 获取保存选项
            save_npy = self.ui.checkBox_3.isChecked()
            if self.index_1 == 0:
                # 执行TV重建
                QtWidgets.QMessageBox.information(self, "信息", "FFT变换...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                # 这里调用TV算法
                results = FFtTransform.process_kspace_files(self.FFT_input_files, self.output_folder_1, save_npy=save_npy)
            elif self.index_1 == 1:
                if not self.traj_path:
                    QtWidgets.QMessageBox.warning(self, "警告", "请先选择输入traj文件！")
                    return
                # 执行WaveletISTA重建
                QtWidgets.QMessageBox.information(self, "信息", "NUFFT变换...")
                # 设置等待光标
                QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                self.setEnabled(False)  # 禁用界面防止用户操作
                results =NUFFtTransform.process_kspace_files(self.FFT_input_files, self.traj_path, self.output_folder_1, save_npy=save_npy)

            if results:
                QtWidgets.QMessageBox.information(self, "完成", "图像重建完成！")

                # 可视化第一个有效结果作为示例
                valid_results = [result for result in results if result is not None]
                print(f"成功处理 {len(valid_results)} 个文件")
                # 显示所有有效结果
                self.show_multiple_results_1(valid_results)

            else:
                QtWidgets.QMessageBox.warning(self, "警告", "重建过程中出现错误！")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"发生错误: {str(e)}")
        finally:
            # 无论成功失败都恢复光标和界面状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def select_input_files_FFT(self):
        """选择多个kspace文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择kspace文件（可多选）",
            "",
            "Numpy Files (*.npy)"
        )

        if files:
            self.FFT_input_files = files  # 存储为文件列表

            # 显示选中的文件数量，并用省略号显示部分路径
            if len(files) == 1:
                display_text = f"输入文件 (1个):\n{files[0]}"
            else:
                display_text = f"输入文件 ({len(files)}个):\n{files[0]} 等..."

            elided_text = self.ui.label_3.fontMetrics().elidedText(
                display_text,
                QtCore.Qt.ElideMiddle,
                self.ui.label_3.width() - 10,
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_3.setText(elided_text)
            self.ui.label_3.setToolTip("\n".join(files))  # 悬停时显示所有完整路径



    def select_traj_file_1(self):
        """选择mask文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择mask文件", "", "Numpy Files (*.npy)")
        if file:
            self.traj_path = file
            # 使用省略号显示部分路径
            elided_text = self.ui.label_4.fontMetrics().elidedText(
                f"输入文件:\n{file}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_4.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_4.setText(elided_text)
            self.ui.label_4.setToolTip(file)  # 悬停时显示完整路径

    def set_algorithm_type_1(self, index=0):
        self.index_1 = index  # 更新算法类型

    def show_multiple_results_1(self, valid_results, max_cols=3):
        num_results = len(valid_results)
        if num_results == 0:
            return

        # 清除之前的图形
        self.ui.fig_1.clear()

        # 计算行数和列数
        cols = min(max_cols, num_results)
        rows = (num_results + cols - 1) // cols

        # 创建子图
        if num_results > 1:
            axes = self.ui.fig_1.subplots(rows, cols)
            axes = axes.ravel()  # 展平axes数组方便迭代
        else:
            axes = [self.ui.fig_1.add_subplot(111)]  # 单个结果的情况

        # 绘制每个结果
        for i, (reconstructed) in enumerate(valid_results):
            ax = axes[i]
            ax.imshow(reconstructed, cmap='gray')
            ax.set_title(f"Recon {i + 1}")
            ax.axis('off')

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # 调整子图间距
        self.ui.fig_1.tight_layout()

        # 重绘画布
        self.ui.canvas_1.draw()

##**====================================================================================================================**##
#深度学习
    def run_inference_and_display(self):
        """
        执行推理，并在 UI 的 Matplotlib 画布上显示 recon vs undersampled
        同时保存图像到用户选择的文件夹

        Args:
            self: 主窗口实例
            config_path: 配置文件路径
            checkpoint_path: 模型权重路径
            max_samples: 最多显示/保存几组
        """
        if self.index_2 == 0:
            results = self.get_UNet_results()

        elif self.index_2 == 1:
            results = self.get_MoDL_results()

        elif self.index_2 == 2:
            results = self.get_Transformer_results()

        elif self.index_2 == 3:
            results = self.get_ISTA_results()

       # 无论成功失败都恢复光标和界面状态
        QApplication.restoreOverrideCursor()
        self.setEnabled(True)
        # 随机选最多 max_samples 组
        selected = results
        if len(selected) > self.max_samples:
            # selected = random.sample(results, self.max_samples) #随机选择self.max_samples=3 组数据
            selected = results[:self.max_samples]  #选择前self.max_samples=3组数据

        # 清除旧画布内容
        self.ui.fig_page_3.clear()

        #  绘图（每组图像一行：undersampled + recon）
        n = len(selected)
        if n == 0:
            QtWidgets.QMessageBox.warning(self, "警告", "没有数据可用于显示。")
            return

        for idx, result in enumerate(selected):
            # 第一行：undersampled
            ax1 = self.ui.fig_page_3.add_subplot(n, 2, 2 * idx + 1)
            ax1.imshow(result['undersampled'], cmap='gray')
            ax1.set_title(f"US-{result['base_name']}")
            ax1.axis('off')

            # 第二行：reconstructed
            ax2 = self.ui.fig_page_3.add_subplot(n, 2, 2 * idx + 2)
            ax2.imshow(result['recon'], cmap='gray')
            ax2.set_title(f"Re-{result['base_name']}")
            ax2.axis('off')

        # self.setEnabled(True)
        # self.ui.fig_page_3.tight_layout()
        self.ui.canvas_page_3.draw()


    def get_UNet_results(self):

        if not hasattr(self, 'input_folder_DL') or not self.input_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return
        if not hasattr(self, 'output_folder_DL') or not self.output_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return

            # 创建数据集
        try:
            self.dataset = UNetInference.DenoisingDataset(root_dir=self.input_folder_DL, image_size=256)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True,
                                                            num_workers=0,
                                                            pin_memory=False)
            QMessageBox.information(self, "成功", "数据集导入成功，开始重建!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集加载失败:\n{str(e)}")
            return False

        # Step 1: 加载配置
        try:
            model_name = 'U-Net'
            config_loader = UNetInference.InferenceConfigLoader(self.config_path, model_name)
            config = config_loader.config

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载配置失败：\n{str(e)}")
            return

        # Step 2: 加载模型
        try:
            model = unet.UNet(
                in_channels=config.get("IN_CHANNELS", 1),
                out_channels=config.get("OUT_CHANNELS", 1)
            ).to(config["DEVICE"])
            model = UNetInference.load_model(model, config, self.checkpoint_path_UNet)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载模型失败：\n{str(e)}")
            return

        # Step 5: 推理并收集结果
        results = []
        model.eval()
        device = next(model.parameters()).device
        # 设置等待光标
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.setEnabled(False)  # 禁用界面防止用户操作

        with torch.no_grad():

            for batch_idx, (noisy_images, base_names) in enumerate(self.dataloader):
                print(f"Processing batch {batch_idx}, names: {base_names}")  # 加日志

                try:
                    noisy_images = noisy_images.to(device)

                    outputs = model(noisy_images)

                    for i in range(outputs.shape[0]):

                        recon_img = UNetInference.normalize_image(outputs[i].cpu().numpy())
                        undersampled_img = UNetInference.normalize_image(noisy_images[i].cpu().numpy())
                        base_name = base_names[i]

                        save_path = os.path.join(self.output_folder_DL, f"recon-{base_name}.png")
                        # 保存PNG图像
                        recon_pil = Image.fromarray((recon_img * 255).astype(np.uint8))
                        recon_pil.save(
                            os.path.join(self.output_folder_DL, save_path))
                        # 保存npy格式
                        save_npy = self.ui.checkBox_2.isChecked()
                        if save_npy:
                            np.save(os.path.join(self.output_folder_DL, f"recon-{base_name}.npy"), recon_img)

                        results.append({
                            'recon': recon_img,
                            'undersampled': undersampled_img,
                            'base_name': base_name
                        })

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        return results

    def get_MoDL_results(self):
        # 检查输入输出文件夹

        if not hasattr(self, 'input_folder_DL') or not self.input_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return False
        if not hasattr(self, 'output_folder_DL') or not self.output_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return False

        # 创建数据集
        try:
            self.dataset = MoDLInference.CustomCSMRIDataset(root_dir=self.input_folder_DL, image_size=256)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True,
                                                              num_workers=0,
                                                              pin_memory=False)
            QMessageBox.information(self, "成功", "数据集导入成功,开始重建!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集加载失败:\n{str(e)}")
            return False


        try:
            # 设置等待状态
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.setEnabled(False)

            # Step 2: 加载配置和模型
            try:
                model_name = 'MoDL'
                config_loader = MoDLInference.InferenceConfigLoader(self.config_path, model_name)
                config = config_loader.config
                optim_config = config.get("optim", {})

                # 初始化模型
                model = MoDL.MoDLRecon(
                    num_iters=optim_config.get("num_iters", 8),
                    channels=optim_config.get("channels", 128)
                )

                # 加载预训练权重
                model = MoDLInference.load_model(model, config, self.checkpoint_path_MoDL)
                device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
                model = model.to(device)
                model.eval()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
                return False

            # Step 3: 推理处理
            results = []
            try:
                with torch.no_grad():
                    for batch_idx, (y, mask, base_names) in enumerate(self.dataloader):
                        print(f"Processing batch {batch_idx}, names: {base_names}")

                        # 确保数据在正确设备上
                        y = y.to(device, dtype=torch.complex64)
                        mask = mask.to(device)

                        # 执行推理
                        outputs = model(y, mask).clamp(0, 1)

                        # 处理每个样本
                        for i in range(outputs.shape[0]):
                            try:
                                # 获取基础名称
                                base_name = base_names[i]

                                # 处理重建图像
                                recon_img = MoDLInference.normalize_image(outputs[i].cpu().numpy())

                                # 处理降采样图像
                                undersampled_k = y[i].cpu().numpy()
                                undersampled_img = MoDLInference.normalize_image(np.abs(np.fft.ifft2(undersampled_k)))

                                # 保存结果
                                save_path = os.path.join(self.output_folder_DL, f"recon-{base_name}.png")
                                recon_pil = Image.fromarray((recon_img * 255).astype(np.uint8))
                                recon_pil.save(save_path)

                                # 可选保存npy
                                if self.ui.checkBox_2.isChecked():
                                    np.save(os.path.join(self.output_folder_DL, f"recon-{base_name}.npy"), recon_img)

                                results.append({
                                    'recon': recon_img,
                                    'undersampled': undersampled_img,
                                    'base_name': base_name
                                })

                            except Exception as e:
                                print(f"Error processing sample {base_name}: {str(e)}")
                                continue
            #
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    QMessageBox.critical(self, "错误", "GPU内存不足! 请尝试减小batch size")
                else:
                    QMessageBox.critical(self, "错误", f"推理过程中出错:\n{str(e)}")
                return False

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理过程中出错:\n{str(e)}")
            return False

        finally:
            # 恢复UI状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        QMessageBox.information(self, "成功", "MoDL重建完成!")
        return results

    def get_Transformer_results(self):
        # 检查输入输出文件夹

        if not hasattr(self, 'input_folder_DL') or not self.input_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return False
        if not hasattr(self, 'output_folder_DL') or not self.output_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return False

        # 创建数据集
        try:
            self.dataset = TransformerInference.CustomCSMRIDataset(root_dir=self.input_folder_DL, image_size=256)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True,
                                                              num_workers=0,
                                                              pin_memory=False)
            QMessageBox.information(self, "成功", "数据集导入成功,开始重建!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集加载失败:\n{str(e)}")
            return False


        try:
            # 设置等待状态
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.setEnabled(False)

            # Step 2: 加载配置和模型
            try:
                model_name = 'Transformer'
                config_loader = TransformerInference.InferenceConfigLoader(self.config_path, model_name)
                config = config_loader.config
                device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
                # 初始化模型
                model = Transformer.HybridTransformer(config)
                # 加载预训练权重
                model = TransformerInference.load_model(model, config, self.checkpoint_path_Transformer)
                model = model.to(device)
                model.eval()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
                return False

            # Step 3: 推理处理
            results = []
            try:
                with torch.no_grad():
                    for batch_idx, (y, mask, base_names) in enumerate(self.dataloader):
                        print(f"Processing batch {batch_idx}, names: {base_names}")

                        # 确保数据在正确设备上
                        y = y.to(device, dtype=torch.complex64)
                        mask = mask.to(device)

                        # 执行推理
                        outputs = model(y, mask).clamp(0, 1)

                        # 处理每个样本
                        for i in range(outputs.shape[0]):
                            try:
                                # 获取基础名称
                                base_name = base_names[i]

                                # 处理重建图像
                                recon_img = TransformerInference.normalize_image(outputs[i].cpu().numpy())

                                # 处理降采样图像
                                undersampled_k = y[i].cpu().numpy()
                                undersampled_img = TransformerInference.normalize_image(np.abs(np.fft.ifft2(undersampled_k)))

                                # 保存结果
                                save_path = os.path.join(self.output_folder_DL, f"recon-{base_name}.png")
                                recon_pil = Image.fromarray((recon_img * 255).astype(np.uint8))
                                recon_pil.save(save_path)

                                # 可选保存npy
                                if self.ui.checkBox_2.isChecked():
                                    np.save(os.path.join(self.output_folder_DL, f"recon-{base_name}.npy"), recon_img)

                                results.append({
                                    'recon': recon_img,
                                    'undersampled': undersampled_img,
                                    'base_name': base_name
                                })

                            except Exception as e:
                                print(f"Error processing sample {base_name}: {str(e)}")
                                continue
            #
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    QMessageBox.critical(self, "错误", "GPU内存不足! 请尝试减小batch size")
                else:
                    QMessageBox.critical(self, "错误", f"推理过程中出错:\n{str(e)}")
                return False

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理过程中出错:\n{str(e)}")
            return False

        finally:
            # 恢复UI状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        QMessageBox.information(self, "成功", "Transformer重建完成!")
        return results

    def get_ISTA_results(self):
        # 检查输入输出文件夹

        if not hasattr(self, 'input_folder_DL') or not self.input_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输入文件夹！")
            return False
        if not hasattr(self, 'output_folder_DL') or not self.output_folder_DL:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹！")
            return False

        # 创建数据集
        try:
            self.dataset = ISTAInference.CustomCSMRIDataset(root_dir=self.input_folder_DL, image_size=256)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True,
                                                              num_workers=0,
                                                              pin_memory=False)
            QMessageBox.information(self, "成功", "数据集导入成功,开始重建!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集加载失败:\n{str(e)}")
            return False


        try:
            # 设置等待状态
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.setEnabled(False)

            # Step 2: 加载配置和模型
            try:
                model_name = 'ISTA'
                config_loader = ISTAInference.InferenceConfigLoader(self.config_path, model_name)
                config = config_loader.config
                device = torch.device(config.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
                # 初始化模型
                model = ista.ISTANetRecon(
                    num_iters=config["optim"]["num_iters"],
                    channels=config["optim"]["channels"]
                )
                # 加载预训练权重
                model = ISTAInference.load_model(model, config, self.checkpoint_path_ISTA)
                model = model.to(device)
                model.eval()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")
                return False

            # Step 3: 推理处理
            results = []
            try:
                with torch.no_grad():
                    for batch_idx, (y, mask, base_names) in enumerate(self.dataloader):
                        print(f"Processing batch {batch_idx}, names: {base_names}")

                        # 确保数据在正确设备上
                        y = y.to(device, dtype=torch.complex64)
                        mask = mask.to(device)

                        # 执行推理
                        outputs = model(y, mask).clamp(0, 1)

                        # 处理每个样本
                        for i in range(outputs.shape[0]):
                            try:
                                # 获取基础名称
                                base_name = base_names[i]

                                # 处理重建图像
                                recon_img = ISTAInference.normalize_image(outputs[i].cpu().numpy())

                                # 处理降采样图像
                                undersampled_k = y[i].cpu().numpy()
                                undersampled_img = ISTAInference.normalize_image(
                                    np.abs(np.fft.ifft2(undersampled_k)))

                                # 保存结果
                                save_path = os.path.join(self.output_folder_DL, f"recon-{base_name}.png")
                                recon_pil = Image.fromarray((recon_img * 255).astype(np.uint8))
                                recon_pil.save(save_path)

                                # 可选保存npy
                                if self.ui.checkBox_2.isChecked():
                                    np.save(os.path.join(self.output_folder_DL, f"recon-{base_name}.npy"), recon_img)

                                results.append({
                                    'recon': recon_img,
                                    'undersampled': undersampled_img,
                                    'base_name': base_name
                                })

                            except Exception as e:
                                print(f"Error processing sample {base_name}: {str(e)}")
                                continue
            #
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    QMessageBox.critical(self, "错误", "GPU内存不足! 请尝试减小batch size")
                else:
                    QMessageBox.critical(self, "错误", f"推理过程中出错了:\n{str(e)}")
                return False

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理过程中出错:\n{str(e)}")
            return False

        finally:
            # 恢复UI状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        QMessageBox.information(self, "成功", "ISTA重建完成!")
        return results

    def set_inference_model(self,index):
        self.index_2 = index

    def select_input_folder_DL(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            # 检查子文件夹是否存在
            required_folders = ['undersampled', 'mask']
            missing_folders = []

            for subfolder in required_folders:
                if not os.path.exists(os.path.join(folder, subfolder)):
                    missing_folders.append(subfolder)

            if missing_folders:
                QMessageBox.warning(self, "Error",
                                    f"未找到所需的子文件夹:\n{', '.join(missing_folders)}")
                return False
            else:
                self.input_folder_DL = folder

                ## 使用省略号显示部分路径
                elided_text = self.ui.label_10.fontMetrics().elidedText(
                    f"输入文件:\n{folder}",
                    QtCore.Qt.ElideMiddle,
                    self.ui.label_10.width() - 10,  # 保留一些边距
                    QtCore.Qt.TextShowMnemonic
                )
                self.ui.label_10.setText(elided_text)
                self.ui.label_10.setToolTip(folder)  # 悬停时显示完整路径
                return True  # 返回 True 表示成功
        return False  # 如果没有选择文件夹，也返回 False

    def select_output_folder_DL(self):
        folder = QFileDialog.getExistingDirectory(self, "请选择输出文件夹")
        if folder:

            self.output_folder_DL = folder
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_11.fontMetrics().elidedText(
                f"输入文件:\n{folder}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_11.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
                )
            self.ui.label_11.setText(elided_text)
            self.ui.label_11.setToolTip(folder)  # 悬停时显示完整路径

##**===================================================================================================================**##
#数据处理
    def create_dataset_structure(self):  #创建数据集
        """创建数据集文件夹结构的完整流程"""
        # 1. 让用户选择父文件夹
        folder = QFileDialog.getExistingDirectory(
            self, "选择数据集父文件夹", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not folder:  # 用户取消了选择
            self.ui.label_15.setText("操作已取消")
            return

        # 2. 定义要创建的文件夹结构
        folders = [
            "data/train/full",
            "data/train/undersampled",
            "data/train/mask",
            "data/train/traj",
            "data/val/full",
            "data/val/undersampled",
            "data/val/mask",
            "data/val/traj"
        ]

        base_path = Path(folder)
        self.ui.label_15.setText(f"正在创建文件夹结构...")
        QApplication.processEvents()  # 更新UI

        try:
            # 3. 创建每个文件夹
            created_folders = []
            for folder in folders:
                full_path = base_path / folder
                full_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(str(full_path.relative_to(base_path)))

            # 4. 显示结果
            success_msg = "数据集文件夹结构创建成功！\n\n创建了以下文件夹:\n"
            success_msg += "\n".join(created_folders)
            QMessageBox.information(self, "成功", success_msg)
            self.ui.label_15.setText(f"创建完成于: {base_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建文件夹时出错:\n{str(e)}")
            self.ui.label_15.setText("创建失败")

    def select_fid_files(self):  #加载fid文件
        """选择加载的fid文件"""
        file, _ = QFileDialog.getOpenFileName(
        None,
        "选择FID数据文件",
        "*.fid;*.FID"
    )
        if file:
            self.input_fid_files = file
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_25.fontMetrics().elidedText(
                f"输入文件:\n{file}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_25.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_25.setText(elided_text)
            self.ui.label_25.setToolTip(file)  # 悬停时显示完整路径


    def execute_transform(self):  #执行转换
        # 1. 交互式选择输入文件
        input_file = self.input_fid_files
        if not input_file:
            print("未选择输入文件，程序终止")
            return
        Fid2Img.execute_transform(input_file)


    def select_img_folder(self):
        """加载数据目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if dir_path:
            self.input_img_folder = dir_path
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_28.fontMetrics().elidedText(
                f"输入文件:\n{dir_path}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_28.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_28.setText(elided_text)
            self.ui.label_28.setToolTip(dir_path)  # 悬停时显示完整路径
            # 自动设置输出目录为输入目录下的split_output
            self.output_img_dir = os.path.join(dir_path, "split_output")

    def execute_split(self):
        """执行分组操作"""
        if not self.input_img_folder:
            QMessageBox.warning(self, "警告", "请先加载数据文件夹")
            return

        # 获取分组数量
        num_parts, ok = QInputDialog.getInt(
            self, "设置分组数量", "请输入要分成的组数:",
            value=self.num_parts, min=1, max=100, step=1
        )

        if not ok:
            return  # 用户取消了输入

        self.num_parts = num_parts

        # 检查输出目录是否存在
        if os.path.exists(self.output_img_dir):
            reply = QMessageBox.question(
                self, "目录已存在",
                f"输出目录 {self.output_img_dir} 已存在，是否覆盖?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            try:
                shutil.rmtree(self.output_img_dir)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法删除旧目录: {str(e)}")
                return

        # 执行分组
        try:
            ImgSplit.split_images_into_parts(self.input_img_folder, self.output_img_dir, self.num_parts)
            QMessageBox.information(self, "完成", f"图片已成功分成 {self.num_parts} 组!")
            self.ui.label_28.setText(f"分组完成，共分成 {self.num_parts} 组")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分组过程中出错: {str(e)}")
            self.ui.label_28.setText("分组失败")

    def plot_mask(self, mask):
        if not hasattr(self, '_ax_page_9'):
            self._ax_page_9 = self.ui.fig_page_9.add_subplot(111)
        ax = self._ax_page_9
        ax.clear()
        ax.imshow(abs(mask), cmap="gray")
        ax.set_title("Mask")
        ax.axis('off')
        self.ui.canvas_page_9.draw()

    def plot_traj(self,traj):
        if not hasattr(self, '_ax_page_9'):
            self._ax_page_9 = self.ui.fig_page_9.add_subplot(111)
        ax = self._ax_page_9
        ax.clear()
        # 采样轨迹
        ax.scatter(traj[:, 0], traj[:, 1], s=1, c="b")
        ax.set_title('SamplingTrajectory')
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.grid(True)
        ax.set_aspect('equal')  # 强制x:y = 1:1
        self.ui.canvas_page_9.draw()


    def visual(self):
        try:
            if self.index_combo_4 == 0:
                params = self.get_phase_params()
                print("Phase Params:", params)  # 检查参数是否正确
                mask = mask_2d.mask2d_phase(image_size=params["arg1"],radius=params["arg2"], undersampling=params["arg3"],axis=params["arg4"])
                self.plot_mask(mask)

            elif self.index_combo_4 == 1:
                params = self.get_position_params()
                print("Position Params:", params)  # 检查参数是否正确
                mask = mask_2d.mask2d_center(image_size=params["arg1"],center_r=params["arg2"], undersampling=params["arg3"])
                self.plot_mask(mask)

            elif self.index_combo_4 == 2:
                params = self.get_radial_params()
                print("Radial Params:",params)
                mask = mask_2d.mask2d_radial(image_size=params["arg1"],num_rays=params["arg2"], undersampling=params["arg3"])
                self.plot_mask(mask)

            elif self.index_combo_4 == 3:
                params = self.get_spiral_params()
                print("Spiral Params:", params)
                mask = mask_2d.mask2d_spiral(image_size=params["arg1"],num_turns=params["arg2"], undersampling=params["arg3"],spacing_power=params["arg4"])
                self.plot_mask(mask)

            elif self.index_combo_4 == 4:
                params = self.get_radial_params_2()
                print("Radial Params(traj):", params)
                traj = mask_2d.traj2d_radial(image_size=params["arg1"],num_rays=params["arg2"], undersampling=params["arg3"])
                self.plot_traj(traj)

            elif self.index_combo_4 == 5:
                params = self.get_spiral_params_2()
                print("Spiral Params(traj):", params)
                traj = mask_2d.traj2d_spiral(image_size=params["arg1"],num_turns=params["arg2"], undersampling=params["arg3"],spacing_power=params["arg4"])
                self.plot_traj(traj)


        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{str(e)}")


    def set_sampling_type(self,index=0):
        self.index_combo_4 = index

    def get_phase_params(self):  #获取mask_phase的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_11, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_11, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_11, 2, 0),
            "arg4": self.get_tablewidget_data(self.ui.tableWidget_11, 3, 0),

        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_position_params(self): #获取mask_center的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_12, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_12, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_12, 2, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_radial_params(self):  # 获取mask_radial的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_13, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_13, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_13, 2, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_spiral_params(self):  # 获取mask_spiral的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_14, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_14, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_14, 2, 0),
            "arg4": self.get_tablewidget_data(self.ui.tableWidget_14, 3, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_radial_params_2(self):  # 获取traj_radial的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_15, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_15, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_15, 2, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params

    def get_spiral_params_2(self):  # 获取traj_spiral的函数的参数
        params = {
            "arg1": self.get_tablewidget_data(self.ui.tableWidget_16, 0, 0),
            "arg2": self.get_tablewidget_data(self.ui.tableWidget_16, 1, 0),
            "arg3": self.get_tablewidget_data(self.ui.tableWidget_16, 2, 0),
            "arg4": self.get_tablewidget_data(self.ui.tableWidget_16, 3, 0),
        }
        # 检查关键参数是否为空
        for key, value in params.items():
            if value is None:
                raise ValueError(f"参数 {key} 不能为空！")

        return params


    def select_name_folder(self):
        input_folder = QFileDialog.getExistingDirectory(
            self, "选择源文件目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if input_folder:
            self.input_name_folder = input_folder
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_28.fontMetrics().elidedText(
                f"输入文件:\n{input_folder}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_28.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_26.setText(elided_text)
            self.ui.label_26.setToolTip(input_folder)  # 悬停时显示完整路径

        output_folder = QFileDialog.getExistingDirectory(
            self, "选择输出目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if output_folder:
            self.output_name_folder = output_folder
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_28.fontMetrics().elidedText(
                f"输入文件:\n{output_folder}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_28.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_26.setText(elided_text)
            self.ui.label_26.setToolTip(output_folder)  # 悬停时显示完整路径



    def get_suffix_and_rename(self):
        suffix, ok = QInputDialog.getText(
            self, "文件名修饰",
            "请输入要添加到文件名的字符串:\n(将添加在文件名和扩展名之间)"
        )
        if ok and suffix:
            self.suffix = suffix
            self.ui.label_26.setText(f"文件名修饰: {suffix}")
            self.batch_rename_files()

    def batch_rename_files(self):
        try:
            if not self.input_name_folder or not self.output_name_folder:
                QMessageBox.warning(self, "警告", "请先选择输入和输出目录")
                return

            # 检查输入文件夹是否存在
            if not os.path.exists(self.input_name_folder):
                QMessageBox.critical(self, "错误", "输入目录不存在")
                return

            # 获取输入目录中的文件列表
            input_files = [f for f in os.listdir(self.input_name_folder) if
                           os.path.isfile(os.path.join(self.input_name_folder, f))]
            file_count = len(input_files)

            if file_count == 0:
                QMessageBox.warning(self, "警告", "输入目录中没有文件")
                return

            # 确认操作
            confirm_msg = f"将在 {file_count} 个文件添加后缀 '{self.suffix}'\n从: {self.input_name_folder}\n到: {self.output_name_folder}"

            reply = QMessageBox.question(
                self, "确认操作", confirm_msg + "\n\n是否继续?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            # 创建输出子文件夹（使用时间戳避免重名）
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"renamed_{self.suffix}"
            output_subfolder = os.path.join(self.output_name_folder, subfolder_name)

            # 创建子文件夹
            os.makedirs(output_subfolder, exist_ok=True)

            # 处理文件
            processed_files = []
            for filename in input_files:
                input_path = os.path.join(self.input_name_folder, filename)

                if os.path.isfile(input_path):
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{name}_{self.suffix}{ext}"
                    output_path = os.path.join(output_subfolder, new_filename)

                    shutil.copy2(input_path, output_path)
                    processed_files.append(new_filename)

            # 显示完成报告
            report = f"处理完成！\n共处理 {len(processed_files)} 个文件\n结果保存在:\n{output_subfolder}\n\n首尾文件示例:\n"
            if processed_files:
                report += f"第一个文件: {processed_files[0]}\n"
                if len(processed_files) > 1:
                    report += f"最后一个文件: {processed_files[-1]}"

            QMessageBox.information(self, "完成", report)

            # 询问是否打开输出子文件夹
            reply = QMessageBox.question(
                self, "查看结果", "是否打开输出子文件夹?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                os.startfile(output_subfolder)  # Windows
                # 对于 macOS: os.system(f"open '{output_subfolder}'")
                # 对于 Linux: os.system(f"xdg-open '{output_subfolder}'")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{str(e)}")


    def set_img_folder(self):
        input_folder = QFileDialog.getExistingDirectory(
            self, "选择源文件目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if input_folder:
            self.input_img_dir = input_folder
            ## 使用省略号显示部分路径
            elided_text = self.ui.label_16.fontMetrics().elidedText(
                f"输入文件:\n{input_folder}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_16.width() - 10,  # 保留一些边距
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_16.setText(elided_text)
            self.ui.label_16.setToolTip(input_folder)  # 悬停时显示完整路径

        output_folder = QFileDialog.getExistingDirectory(
            self, "选择输出目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if output_folder:
            self.output_kspace_dir = os.path.join(output_folder, "undersampled")
            self.output_mask_dir = os.path.join(output_folder, "mask")
            os.makedirs(self.output_kspace_dir, exist_ok=True)
            os.makedirs(self.output_mask_dir, exist_ok=True)

            # 使用省略号显示部分路径
            elided_text = self.ui.label_16.fontMetrics().elidedText(
                f"输出目录:\n{output_folder}",
                QtCore.Qt.ElideMiddle,
                self.ui.label_16.width() - 10,
                QtCore.Qt.TextShowMnemonic
            )
            self.ui.label_16.setText(elided_text)
            self.ui.label_16.setToolTip(output_folder)

    def generate_masks(self):
        """生成kspace 和 mask/traj"""
        if not self.input_img_dir or not self.output_kspace_dir or not self.output_mask_dir:
            QMessageBox.warning(self, "警告", "请先选择输入和输出目录!")
            return

        try:
            # 设置等待状态
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.setEnabled(False)

            for img_file in os.listdir(self.input_img_dir):
                if not img_file.lower().endswith(('.png', '.jpg','.jpeg', '.bmp', '.gif')):
                    continue

                # 加载并预处理图像
                img_path = os.path.join(self.input_img_dir, img_file)
                img = Image.open(img_path).convert('L').resize((self.image_size, self.image_size))  # 转灰度图并调整大小
                x = np.array(img) / 255.0  # 归一化到 [0, 1]
                x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # 转换为 Tensor 并增加通道维度
                # 生成 k-space 数据
                kspace = fft.fftshift(fft.fft2(x_tensor), dim=(-2, -1)) # 计算 k-space 并中心化


                if self.index_combo_4 == 0:
                    params = self.get_phase_params()
                    print("Phase Params:", params)  # 检查参数是否正确
                    mask = mask_2d.mask2d_phase(image_size=params["arg1"], radius=params["arg2"],
                                                undersampling=params["arg3"], axis=params["arg4"])
                    undersampled_kspace = kspace.numpy() * mask

                elif self.index_combo_4 == 1:
                    params = self.get_position_params()
                    print("Position Params:", params)  # 检查参数是否正确
                    mask = mask_2d.mask2d_center(image_size=params["arg1"], center_r=params["arg2"],
                                                 undersampling=params["arg3"])
                    undersampled_kspace = kspace.numpy() * mask

                elif self.index_combo_4 == 2:
                    params = self.get_radial_params()
                    print("Radial Params:", params)
                    mask = mask_2d.mask2d_radial(image_size=params["arg1"], num_rays=params["arg2"],
                                                 undersampling=params["arg3"])
                    undersampled_kspace = kspace.numpy() * mask

                elif self.index_combo_4 == 3:
                    params = self.get_spiral_params()
                    print("Spiral Params:", params)
                    mask = mask_2d.mask2d_spiral(image_size=params["arg1"], num_turns=params["arg2"],
                                                 undersampling=params["arg3"], spacing_power=params["arg4"])
                    undersampled_kspace = kspace.numpy() * mask


                elif self.index_combo_4 == 4:
                    params = self.get_radial_params_2()
                    print("Radial Params(traj):", params)
                    mask = mask_2d.traj2d_radial(image_size=params["arg1"], num_rays=params["arg2"],
                                                 undersampling=params["arg3"])

                    undersampled_kspace = mask_2d.kspace_nufft(x, mask)

                elif self.index_combo_4 == 5:
                    params = self.get_spiral_params_2()
                    print("Spiral Params(traj):", params)
                    mask = mask_2d.traj2d_spiral(image_size=params["arg1"], num_turns=params["arg2"],
                                                 undersampling=params["arg3"], spacing_power=params["arg4"])

                    undersampled_kspace = mask_2d.kspace_nufft(x, mask)

                # 保存欠采样 k-space 数据
                kspace_output_path = os.path.join(self.output_kspace_dir, img_file.replace('.png', '_kspace.npy'))
                np.save(kspace_output_path, undersampled_kspace)

                # 保存 mask
                mask_output_path = os.path.join(self.output_mask_dir, img_file.replace('.png', '_mask.npy'))
                np.save(mask_output_path, mask)

            # 恢复UI状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

            QMessageBox.information(self, "完成", "Underkspace与Mask生成完成!")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{str(e)}")

    def select_input_folder_3(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            # 检查子文件夹是否存在
            required_folders = ['train/full', 'train/undersampled', 'train/mask']
            missing_folders = []

            for subfolder in required_folders:
                if not os.path.exists(os.path.join(folder, subfolder)):
                    missing_folders.append(subfolder)

            if missing_folders:
                QMessageBox.warning(self, "Error",
                                    f"未找到所需的子文件夹:\n{', '.join(missing_folders)}")
                return False
            else:
                self.input_base_path = folder
                ## 使用省略号显示部分路径
                elided_text = self.ui.label_29.fontMetrics().elidedText(
                    f"输入文件:\n{folder}",
                    QtCore.Qt.ElideMiddle,
                    self.ui.label_29.width() - 10,  # 保留一些边距
                    QtCore.Qt.TextShowMnemonic
                )
                self.ui.label_29.setText(elided_text)
                self.ui.label_29.setToolTip(folder)  # 悬停时显示完整路径
                QMessageBox.information(self, "Success",
                                        "输入文件夹结构验证成功!")
                return True  # 返回 True 表示成功
        return False  # 如果没有选择文件夹，也返回 False

    def execute_transfer(self):
        # 如果选择输入文件夹失败，直接返回
        if not self.select_input_folder_3():
            return
        # 获取输出文件夹
        output_folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not output_folder:
            return

        self.output_base_path = output_folder
        ## 使用省略号显示部分路径
        elided_text = self.ui.label_29.fontMetrics().elidedText(
            f"输入文件:\n{output_folder}",
            QtCore.Qt.ElideMiddle,
            self.ui.label_29.width() - 10,  # 保留一些边距
            QtCore.Qt.TextShowMnemonic
        )
        self.ui.label_29.setText(elided_text)
        self.ui.label_29.setToolTip(output_folder)  # 悬停时显示完整路径

        # 如果不存在所需的验证文件夹，创建
        val_folders = {
            'full': os.path.join(output_folder, 'val/full'),
            'undersampled': os.path.join(output_folder, 'val/undersampled'),
            'mask': os.path.join(output_folder, 'val/mask')
        }

        for folder in val_folders.values():
            os.makedirs(folder, exist_ok=True)

        #设置路径
        input_dir = os.path.join(self.input_base_path, 'train/full')
        kspace_dir = os.path.join(self.input_base_path, 'train/undersampled')
        mask_dir = os.path.join(self.input_base_path, 'train/mask')

        val_input_dir = val_folders['full']
        val_kspace_dir = val_folders['undersampled']
        val_mask_dir = val_folders['mask']

        # 执行转移
        try:
            self.split_to_validation(input_dir, kspace_dir, mask_dir,
                                     val_input_dir, val_kspace_dir, val_mask_dir)
            QMessageBox.information(self, "Success", "文件成功转移!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"发生错误:\n{str(e)}")

    def split_to_validation(self, input_dir, kspace_dir, mask_dir, val_input_dir, val_kspace_dir, val_mask_dir):
        """
        从输入和输出文件夹中，随机选择25%的数据进行验证，
            维护文件名的一致性。
        """
        # Get all files in input directory
        input_files = [f for f in os.listdir(input_dir) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        total_files = len(input_files)

        if total_files == 0:
            raise ValueError("在输入目录中没有找到文件")


        self.validation_ratio = float(1 - float(self.ui.lineEdit.text()))

        # 计算验证集文件数
        num_val_files = int(total_files * self.validation_ratio)

        # 为验证集随机选择文件
        val_input_files = random.sample(input_files, num_val_files)

        # 将选择的文件移到验证集
        for input_file_name in val_input_files:
            # 构造输入文件路径
            input_src_path = os.path.join(input_dir, input_file_name)
            input_dst_path = os.path.join(val_input_dir, input_file_name)

            # 提取基本文件名
            base_name = os.path.splitext(input_file_name)[0]

            # 构造输入kspace文件路径
            kspace_file_name = f"{base_name}_kspace.npy"
            kspace_src_path = os.path.join(kspace_dir, kspace_file_name)
            kspace_dst_path = os.path.join(val_kspace_dir, kspace_file_name)

            # 构造输入mask文件路径
            mask_file_name = f"{base_name}_mask.npy"
            mask_src_path = os.path.join(mask_dir, mask_file_name)
            mask_dst_path = os.path.join(val_mask_dir, mask_file_name)

            # 如果文件存在，则移动文件
            if os.path.exists(input_src_path):
                shutil.move(input_src_path, input_dst_path)
            else:
                print(f"输入文件未找到: {input_file_name}")

            if os.path.exists(kspace_src_path):
                shutil.move(kspace_src_path, kspace_dst_path)
            else:
                print(f"k-space 文件未找到: {kspace_file_name}")

            if os.path.exists(mask_src_path):
                shutil.move(mask_src_path, mask_dst_path)
            else:
                print(f"Mask 文件未找到: {mask_file_name}")

    def trainer(self):
        """开始训练，选择对应的模型"""

        try:

            if self.index_3 == 0:
                parser = argparse.ArgumentParser(description="MRI Denoising Trainer")
                parser.add_argument("--config", type=str, default="./DL_params.json",
                                    help="Path to configuration JSON file")
                parser.add_argument("--model", type=str, default="U-Net",
                                    help="Model name in config file")
                args = parser.parse_args()
                # 初始化训练器
                trainer = unet.MRIDenoisingTrainer(args.config, args.model)

            elif self.index_3 == 1:
                # 初始化训练器
                trainer = MoDL.MoDLTrainer("DL_params.json")
                trainer.load_datasets()

            elif self.index_3 == 2:
                # 初始化训练器
                trainer = Transformer.TransformerTrainer("DL_params.json", model_name="Transformer")
                trainer.load_datasets()

            elif self.index_3 == 3:
                # 初始化训练器
                trainer = ista.Trainer("DL_params.json", model_name="ISTA")
                # load dataset
                trainer.get_dataloaders()

            # 设置等待状态
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.setEnabled(False)
            # 开始训练
            trainer.train()


            # 进行评估
            try:
                evaluation_results = trainer.evaluate()
                self.plot_random_results(evaluation_results, num_groups=3)

                # 更新label显示
                elided_text = f"验证集平均峰值信噪比是：{evaluation_results['average_psnr']};平均结构相似性指数是：{evaluation_results['average_ssim']}"

                self.ui.label_24.setText(elided_text)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"评估失败: {str(e)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"发生错误: {str(e)}")

        finally:
            # 无论成功失败都恢复光标和界面状态
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def plot_random_results(self, evaluation_results, num_groups=3):
        """随机选择num_groups组结果，每组显示ground truth, noisy和reconstructed图像"""
        results = evaluation_results['results']

        # 随机选择3组结果
        if len(results) > num_groups:
            selected_results = random.sample(results, num_groups)
        else:
            selected_results = results

        # 创建子图 - 每行一组，每组3个图像
        fig, axes = self.ui.fig_page_9.subplots(num_groups, 3)

        # 如果只有一组结果，axes的形状需要调整
        if num_groups == 1:
            axes = axes.reshape(1, -1)
            # 绘制每组结果
            for row, result in enumerate(selected_results):
                # 绘制ground truth
                ax = axes[row, 0]
                ax.imshow(result['ground_truth'], cmap='gray')
                ax.set_title(f"Ground Truth\nPSNR: {result['psnr']:.2f}\nSSIM: {result['ssim']:.4f}")
                ax.axis('off')

                # 绘制noisy图像
                ax = axes[row, 1]
                ax.imshow(result['noisy'], cmap='gray')
                # 计算noisy与gt的PSNR和SSIM
                noisy_psnr = psnr(result['ground_truth'], result['noisy'], data_range=1.0)
                noisy_ssim = ssim(result['ground_truth'], result['noisy'], data_range=1.0)
                ax.set_title(f"Noisy Input\nPSNR: {noisy_psnr:.2f}\nSSIM: {noisy_ssim:.4f}")
                ax.axis('off')

                # 绘制reconstructed图像
                ax = axes[row, 2]
                ax.imshow(result['reconstructed'], cmap='gray')
                ax.set_title(f"Reconstructed\nPSNR: {result['psnr']:.2f}\nSSIM: {result['ssim']:.4f}")
                ax.axis('off')

            # 调整子图间距
            self.ui.fig_page_9.tight_layout()

            # 重绘画布
            self.ui.canvas_page_9.draw()

    def set_train_model(self,index=0):
        self.index_3 = index



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageReconApp()
    window.show()
    sys.exit(app.exec_())