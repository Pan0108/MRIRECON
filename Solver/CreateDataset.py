# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget, QFileDialog, QLabel,
                             QMessageBox)
from PyQt5.QtCore import Qt
from pathlib import Path


class DatasetCreatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集创建工具")
        self.setGeometry(100, 100, 400, 200)

        self.initUI()

    def initUI(self):
        # 主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # 说明标签
        info_label = QLabel("点击下方按钮创建标准数据集文件夹结构")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        # 创建数据集按钮
        self.create_btn = QPushButton("创建数据集")
        self.create_btn.clicked.connect(self.create_dataset_structure)
        layout.addWidget(self.create_btn)

        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        main_widget.setLayout(layout)

    def create_dataset_structure(self):
        """创建数据集文件夹结构的完整流程"""
        # 1. 让用户选择父文件夹
        folder = QFileDialog.getExistingDirectory(
            self, "选择数据集父文件夹", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not folder:  # 用户取消了选择
            self.status_label.setText("操作已取消")
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
        self.status_label.setText(f"正在创建文件夹结构...")
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
            self.status_label.setText(f"创建完成于: {base_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建文件夹时出错:\n{str(e)}")
            self.status_label.setText("创建失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetCreatorApp()
    window.show()
    sys.exit(app.exec_())