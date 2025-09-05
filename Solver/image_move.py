# -*- coding: utf-8 -*-
import os
import random
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget, QFileDialog, QMessageBox)


class DataSplitterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Splitter")
        self.setGeometry(100, 100, 300, 200)

        self.input_base_path = ""
        self.output_base_path = ""

        self.initUI()

    def initUI(self):
        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Create buttons
        self.select_folder_btn = QPushButton("Select Input Folder", self)
        self.select_folder_btn.clicked.connect(self.select_input_folder)

        self.execute_btn = QPushButton("Execute Transfer", self)
        self.execute_btn.clicked.connect(self.execute_transfer)
        self.execute_btn.setEnabled(False)  # Disabled until input folder is selected

        # Add buttons to layout
        layout.addWidget(self.select_folder_btn)
        layout.addWidget(self.execute_btn)

        # Set layout
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            # Check if required subfolders exist
            required_folders = ['train/full', 'train/undersampled', 'train/mask']
            missing_folders = []

            for subfolder in required_folders:
                if not os.path.exists(os.path.join(folder, subfolder)):
                    missing_folders.append(subfolder)

            if missing_folders:
                QMessageBox.warning(self, "Error",
                                    f"Required subfolders not found:\n{', '.join(missing_folders)}")
                self.execute_btn.setEnabled(False)
            else:
                self.input_base_path = folder
                self.execute_btn.setEnabled(True)
                QMessageBox.information(self, "Success",
                                        "Input folder structure validated successfully!")

    def execute_transfer(self):
        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            return

        self.output_base_path = output_folder

        # Create required validation folders if they don't exist
        val_folders = {
            'full': os.path.join(output_folder, 'val/full'),
            'undersampled': os.path.join(output_folder, 'val/undersampled'),
            'mask': os.path.join(output_folder, 'val/mask')
        }

        for folder in val_folders.values():
            os.makedirs(folder, exist_ok=True)

        # Set paths
        input_dir = os.path.join(self.input_base_path, 'train/full')
        kspace_dir = os.path.join(self.input_base_path, 'train/undersampled')
        mask_dir = os.path.join(self.input_base_path, 'train/mask')

        val_input_dir = val_folders['full']
        val_kspace_dir = val_folders['undersampled']
        val_mask_dir = val_folders['mask']

        # Perform the transfer
        try:
            self.split_to_validation(input_dir, kspace_dir, mask_dir,
                                     val_input_dir, val_kspace_dir, val_mask_dir)
            QMessageBox.information(self, "Success", "Files transferred successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    def split_to_validation(self, input_dir, kspace_dir, mask_dir, val_input_dir, val_kspace_dir, val_mask_dir,
                            validation_ratio=0.25):
        """
        From input and output folders, randomly select 25% of data for validation,
        maintaining filename consistency.
        """
        # Get all files in input directory
        input_files = [f for f in os.listdir(input_dir) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        total_files = len(input_files)

        if total_files == 0:
            raise ValueError("No files found in the input directory.")

        # Calculate number of validation files
        num_val_files = int(total_files * validation_ratio)

        # Randomly select files for validation
        val_input_files = random.sample(input_files, num_val_files)

        # Move selected files to validation folders
        for input_file_name in val_input_files:
            # Construct input file paths
            input_src_path = os.path.join(input_dir, input_file_name)
            input_dst_path = os.path.join(val_input_dir, input_file_name)

            # Extract base filename
            base_name = os.path.splitext(input_file_name)[0]

            # Construct k-space file paths
            kspace_file_name = f"{base_name}_kspace.npy"
            kspace_src_path = os.path.join(kspace_dir, kspace_file_name)
            kspace_dst_path = os.path.join(val_kspace_dir, kspace_file_name)

            # Construct mask file paths
            mask_file_name = f"{base_name}_mask.npy"
            mask_src_path = os.path.join(mask_dir, mask_file_name)
            mask_dst_path = os.path.join(val_mask_dir, mask_file_name)

            # Move files if they exist
            if os.path.exists(input_src_path):
                shutil.move(input_src_path, input_dst_path)
            else:
                print(f"Input file not found: {input_file_name}")

            if os.path.exists(kspace_src_path):
                shutil.move(kspace_src_path, kspace_dst_path)
            else:
                print(f"k-space file not found: {kspace_file_name}")

            if os.path.exists(mask_src_path):
                shutil.move(mask_src_path, mask_dst_path)
            else:
                print(f"Mask file not found: {mask_file_name}")


if __name__ == "__main__":
    app = QApplication([])
    window = DataSplitterApp()
    window.show()
    app.exec_()