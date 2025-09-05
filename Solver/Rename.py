# # -*- coding: utf-8 -*-
# import os
# import shutil
# import tkinter as tk
# from tkinter import filedialog, messagebox, simpledialog
#
#
# def batch_rename_files():
#     """主处理函数：包含完整的文件处理流程"""
#     # 创建隐藏的Tkinter根窗口
#     root = tk.Tk()
#     root.withdraw()
#
#     try:
#         # 1. 选择输入文件夹
#         input_folder = filedialog.askdirectory(
#             title="选择源文件目录",
#             initialdir=os.getcwd()
#         )
#         if not input_folder:
#             messagebox.showwarning("取消", "未选择输入目录")
#             return
#
#         # 2. 选择输出文件夹
#         output_folder = filedialog.askdirectory(
#             title="选择输出目录",
#             initialdir=os.getcwd()
#         )
#         if not output_folder:
#             messagebox.showwarning("取消", "未选择输出目录")
#             return
#
#         # 3. 输入要添加的字符串
#         suffix = simpledialog.askstring(
#             "文件名修饰",
#             "请输入要添加到文件名的字符串:\n(将添加在文件名和扩展名之间)",
#             parent=root
#         )
#         if suffix is None:  # 用户点击取消
#             messagebox.showinfo("取消", "操作已取消")
#             return
#
#         # 4. 显示确认信息
#         file_count = len([f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))])
#         confirm_msg = f"将在 {file_count} 个文件添加后缀 '{suffix}'\n从: {input_folder}\n到: {output_folder}"
#         if not messagebox.askyesno("确认操作", confirm_msg + "\n\n是否继续?"):
#             return
#
#         # 5. 创建输出目录（如果不存在）
#         os.makedirs(output_folder, exist_ok=True)
#
#         # 6. 处理文件
#         processed_files = []
#         for filename in os.listdir(input_folder):
#             input_path = os.path.join(input_folder, filename)
#
#             if os.path.isfile(input_path):
#                 # 分割文件名和扩展名
#                 name, ext = os.path.splitext(filename)
#
#                 # 构建新文件名
#                 new_filename = f"{name}_{suffix}{ext}"
#                 output_path = os.path.join(output_folder, new_filename)
#
#                 # 复制文件（保留原始文件）
#                 shutil.copy2(input_path, output_path)
#                 processed_files.append(new_filename)
#
#         # 7. 显示完成报告
#         report = f"处理完成！\n共处理 {len(processed_files)} 个文件\n首尾文件示例:\n"
#         report += f"第一个文件: {processed_files[0]}\n" if processed_files else ""
#         report += f"最后一个文件: {processed_files[-1]}" if len(processed_files) > 1 else ""
#
#         messagebox.showinfo("完成", report)
#
#         # 8. 可选：打开输出文件夹
#         if messagebox.askyesno("查看结果", "是否打开输出文件夹?"):
#             os.startfile(output_folder)
#
#     except Exception as e:
#         messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
#     finally:
#         root.destroy()
#
#
# if __name__ == '__main__':
#     batch_rename_files()

# -*- coding: utf-8 -*-
import os
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QMessageBox, QInputDialog, QLabel, QLineEdit)
from PyQt5.QtCore import Qt


class FileRenamer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("批量文件重命名工具")
        self.setGeometry(100, 100, 400, 300)

        self.input_folder = ""
        self.output_folder = ""
        self.suffix = ""

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Input folder selection
        self.input_label = QLabel("源文件目录: 未选择")
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)

        self.input_btn = QPushButton("加载数据 - 选择源目录")
        self.input_btn.clicked.connect(self.select_input_folder)
        layout.addWidget(self.input_btn)

        # Output folder selection
        self.output_label = QLabel("输出目录: 未选择")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)

        self.output_btn = QPushButton("选择输出目录")
        self.output_btn.clicked.connect(self.select_output_folder)
        layout.addWidget(self.output_btn)

        # Suffix input
        self.suffix_label = QLabel("文件名修饰: 未设置")
        layout.addWidget(self.suffix_label)

        self.rename_btn = QPushButton("修改文件名")
        self.rename_btn.clicked.connect(self.get_suffix_and_rename)
        self.rename_btn.setEnabled(False)
        layout.addWidget(self.rename_btn)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Add stretch to push everything up
        layout.addStretch()

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择源文件目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.input_folder = folder
            self.input_label.setText(f"源文件目录: {folder}")
            self.update_status()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择输出目录", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"输出目录: {folder}")
            self.update_status()

    def update_status(self):
        if self.input_folder and self.output_folder:
            file_count = len([f for f in os.listdir(self.input_folder)
                              if os.path.isfile(os.path.join(self.input_folder, f))])
            self.status_label.setText(f"找到 {file_count} 个文件准备处理")
            self.rename_btn.setEnabled(True)
        else:
            self.rename_btn.setEnabled(False)

    def get_suffix_and_rename(self):
        suffix, ok = QInputDialog.getText(
            self, "文件名修饰",
            "请输入要添加到文件名的字符串:\n(将添加在文件名和扩展名之间)"
        )
        if ok and suffix:
            self.suffix = suffix
            self.suffix_label.setText(f"文件名修饰: {suffix}")
            self.batch_rename_files()

    def batch_rename_files(self):
        try:
            # Check if folders are selected
            if not self.input_folder or not self.output_folder:
                QMessageBox.warning(self, "警告", "请先选择输入和输出目录")
                return

            # Confirm operation
            file_count = len([f for f in os.listdir(self.input_folder)
                              if os.path.isfile(os.path.join(self.input_folder, f))])
            confirm_msg = f"将在 {file_count} 个文件添加后缀 '{self.suffix}'\n从: {self.input_folder}\n到: {self.output_folder}"

            reply = QMessageBox.question(
                self, "确认操作", confirm_msg + "\n\n是否继续?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            # Create output directory if not exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Process files
            processed_files = []
            for filename in os.listdir(self.input_folder):
                input_path = os.path.join(self.input_folder, filename)

                if os.path.isfile(input_path):
                    # Split filename and extension
                    name, ext = os.path.splitext(filename)

                    # Build new filename
                    new_filename = f"{name}_{self.suffix}{ext}"
                    output_path = os.path.join(self.output_folder, new_filename)

                    # Copy file (preserve original)
                    shutil.copy2(input_path, output_path)
                    processed_files.append(new_filename)

            # Show completion report
            report = f"处理完成！\n共处理 {len(processed_files)} 个文件\n首尾文件示例:\n"
            if processed_files:
                report += f"第一个文件: {processed_files[0]}\n"
            if len(processed_files) > 1:
                report += f"最后一个文件: {processed_files[-1]}"

            QMessageBox.information(self, "完成", report)

            # Optionally open output folder
            reply = QMessageBox.question(
                self, "查看结果", "是否打开输出文件夹?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                os.startfile(self.output_folder)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{str(e)}")


if __name__ == '__main__':
    app = QApplication([])
    window = FileRenamer()
    window.show()
    app.exec_()