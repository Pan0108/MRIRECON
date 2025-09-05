# MRI图像重建平台详细帮助文档

## 项目概述

这是一个功能强大的MRI（磁共振成像）图像重建与数据处理平台，专门为医学影像研究和工程应用设计。即使您没有任何算法背景，也可以通过这个平台轻松完成MRI图像的重建和处理工作。

平台主要包含以下四大功能模块：
1. **传统重建模块** - 基础的图像重建方法
2. **压缩感知重建模块** - 先进的数学优化算法
3. **深度学习重建模块** - 基于人工智能的重建方法
4. **数据处理工具模块** - 数据预处理和管理工具

## 项目文件结构详解

```
GUI_CSMRI/
├── MriRecon.py              # 主程序入口文件（启动整个软件）
├── Ui_ImageRecon2.py        # 界面代码（不要修改）
├── DL_params.json           # 深度学习模型配置文件（参数设置）
├── README.md                # 项目说明文档
├── CLAUDE.md                # Claude开发指南
├── DLCSMRI/                 # 深度学习相关模块（AI算法）
│   ├── unet.py             # U-Net模型训练代码
│   ├── UNetInference.py    # U-Net模型推理代码
│   ├── MoDL.py             # MoDL模型训练代码
│   ├── MoDLInference.py    # MoDL模型推理代码
│   ├── Transformer.py      # Transformer模型训练代码
│   ├── TransformerInference.py # Transformer模型推理代码
│   ├── ista.py             # ISTA-Net模型训练代码
│   └── ISTAInference.py    # ISTA-Net模型推理代码
├── Solver/                  # 传统算法和压缩感知算法模块（数学算法）
│   ├── ADMM_TV.py          # ADMM-TV算法实现
│   ├── ADMM_TV_CUPY.py     # GPU加速的ADMM-TV算法实现
│   ├── Wavelet_ISTA.py     # 小波-ISTA算法实现
│   ├── FFtTransform.py     # FFT变换实现
│   ├── NUFFtTransform.py   # NUFFT变换实现
│   ├── NUFFtWavelet.py     # NUFFt-小波算法实现
│   ├── mask_2d.py          # 2D掩码生成
│   └── ...
├── model/                   # 预训练模型权重文件（AI模型）
│   ├── U_Net.pth           # U-Net模型权重
│   ├── MoDL.pth            # MoDL模型权重
│   ├── Transformer.pth     # Transformer模型权重
│   ├── ISTA.pth            # ISTA-Net模型权重
│   └── ...
└── output_*/                # 各模型输出目录（结果保存）
```

## 软件启动和基本操作

### 启动软件
```bash
python MriRecon.py
```

### 界面导航
软件界面分为四个主要页面，通过左侧导航栏切换：
1. **传统重建** - 基础的傅里叶变换重建
2. **压缩感知重建** - 高级数学优化算法
3. **深度学习重建** - AI人工智能重建
4. **数据处理** - 数据预处理工具

## 第一部分：压缩感知重建算法详解（CSMRI）

### 什么是压缩感知（Compressed Sensing）？
压缩感知是一种革命性的信号采集和重建理论，它允许我们用远少于传统理论要求的采样点来重建高质量的图像。这对于MRI来说意味着可以大幅缩短扫描时间。

### 核心思想（小白也能理解）
想象你要画一幅画，传统方法是把画布上每个点的颜色都精确记录下来。而压缩感知就像一个聪明的画家，只需要记录关键的几个点，就能根据这些点推断出整幅画的全貌。

在MRI中：
- **传统方法**：需要采集完整的k空间数据（大量采样点）
- **压缩感知**：只需要采集部分k空间数据（少量采样点），通过算法重建完整图像

### 算法1：ADMM-TV（交替方向乘子法+全变分）

#### 算法原理（通俗解释）
ADMM-TV结合了两个强大的概念：
1. **ADMM（交替方向乘子法）**：一种优化算法，像一个聪明的侦探，通过分步骤解决复杂问题
2. **TV（全变分）**：假设图像中相邻像素的变化不会太大，保持图像的平滑性

#### 参数说明
- **tv_r（TV正则化参数）**：控制图像平滑程度，值越大图像越平滑（默认5）
- **rho（ADMM惩罚参数）**：控制优化过程的严格程度（默认1）
- **n_iter（迭代次数）**：算法重复计算的次数，越多结果越精确但耗时越长（默认50）
- **step（步长）**：每次计算的步长大小（默认0.5）

#### 适用场景
- 适合1D随机采样掩码
- 重建速度较快
- 对噪声有一定的抑制能力

#### 调用方式
在主程序中通过`ADMM_TV.process_kspace_files()`函数调用

### 算法2：Wavelet-ISTA（小波变换+迭代软阈值算法）

#### 算法原理（通俗解释）
Wavelet-ISTA利用了图像在小波域的稀疏性：
1. **小波变换**：将图像转换到小波域，大部分信息集中在少数系数上
2. **ISTA（迭代软阈值算法）**：通过迭代优化，逐步逼近真实图像

#### 参数说明
- **epsilon（收敛阈值）**：控制算法停止的精度（默认1e-4）
- **n_max（最大迭代次数）**：最大迭代次数（默认50）
- **tol（容差）**：收敛判断的容差（默认1e-4）
- **decfac（衰减因子）**：控制正则化参数的衰减速度（默认0.5）
- **threshold（小波阈值）**：小波系数的阈值（默认5）
- **wavelet（小波类型）**：使用的小波基函数（默认'haar'）
- **level（小波分解层数）**：小波分解的层数（默认2）

#### 适用场景
- 适合具有稀疏特性的图像
- 能保留更多图像细节
- 对纹理丰富的图像效果更好

#### 调用方式
在主程序中通过`Wavelet_ISTA.process_kspace_files()`函数调用

### 算法3：ADMM-TV-CuPy（GPU加速版本）

#### 算法原理
这是ADMM-TV算法的GPU加速版本，利用CuPy库在NVIDIA GPU上进行并行计算，大幅提高处理速度。

#### 适用场景
- 需要快速处理大量数据
- 有NVIDIA GPU硬件支持
- 对处理时间有严格要求

#### 调用方式
在主程序中通过`ADMM_TV_CUPY.process_kspace_files()`函数调用

### 算法4：NUFFt-Wavelet（非均匀FFT+小波）

#### 算法原理
专为处理非笛卡尔采样轨迹（如径向、螺旋）设计，结合了非均匀快速傅里叶变换和小波稀疏性。

#### 适用场景
- 径向采样轨迹
- 螺旋采样轨迹
- 非标准采样模式

#### 调用方式
在主程序中通过`NUFFtWavelet.process_kspace_files()`函数调用

## 第二部分：深度学习重建模块详解

### 什么是深度学习MRI重建？
深度学习MRI重建使用人工智能神经网络来学习从欠采样数据到高质量图像的映射关系。相比传统算法，深度学习方法通常能获得更好的重建质量。

### 模型1：U-Net

#### 模型结构（通俗解释）
U-Net是一种经典的编码器-解码器结构：
1. **编码器**：像一个摄影师，逐步提取图像的高层次特征
2. **解码器**：像一个画家，根据提取的特征重新绘制完整图像
3. **跳跃连接**：像一座桥梁，将低层次的细节信息传递给高层次

#### 模型特点
- 结构简单，易于理解和实现
- 在医学图像处理中表现优秀
- 训练相对稳定

#### 配置参数（DL_params.json中）
```json
{
  "U-Net": {
    "DEVICE": "cpu",           // 运行设备（cpu或cuda）
    "BATCH_SIZE": 8,           // 批次大小
    "NUM_WORKERS": 6,          // 数据加载线程数
    "LEARNING_RATE": 1e-4,     // 学习率
    "WEIGHT_DECAY": 5e-5,      // 权重衰减
    "NUM_EPOCHS": 1000,        // 训练轮数
    "CHECKPOINT_PATH": "./model/U_Net.pth",  // 模型保存路径
    "RESUME_TRAINING": true,   // 是否恢复训练
    "OUTPUT_DIR": "./output_UNET",  // 输出目录
    "IN_CHANNELS": 1,          // 输入通道数
    "OUT_CHANNELS": 1,         // 输出通道数
    "IMAGE_SIZE": 256          // 图像大小
  }
}
```

#### 训练调用
```bash
python DLCSMRI/unet.py
```

#### 推理调用
在主程序中通过`UNetInference`模块调用

### 模型2：MoDL（Model-Based Deep Learning）

#### 模型结构（通俗解释）
MoDL将传统的迭代优化算法展开为深度网络：
1. **物理模型**：保持了MRI物理规律的约束
2. **数据驱动**：利用神经网络学习最优参数
3. **优势**：结合了物理模型的可靠性和深度学习的灵活性

#### 模型特点
- 理论基础扎实
- 重建质量高
- 泛化能力强

#### 配置参数
```json
{
  "MoDL": {
    "DEVICE": "cpu",
    "train": {
      "batch_size": 4,
      "learning_rate": 1e-4,
      "epochs": 1000
    },
    "optim": {
      "num_iters": 8,      // 迭代次数
      "channels": 256,     // 通道数
      "lambda_init": 0.1   // 初始正则化参数
    }
  }
}
```

#### 训练调用
```bash
python DLCSMRI/MoDL.py
```

#### 推理调用
在主程序中通过`MoDLInference`模块调用

### 模型3：Transformer

#### 模型结构（通俗解释）
Transformer使用自注意力机制：
1. **自注意力**：能够关注图像中所有位置之间的关系
2. **全局建模**：可以捕捉长距离依赖关系
3. **并行处理**：计算效率高

#### 模型特点
- 捕捉全局信息能力强
- 适合处理复杂纹理
- 计算资源需求较大

#### 配置参数
```json
{
  "Transformer": {
    "DEVICE": "cpu",
    "BATCH_SIZE": 4,
    "model": {
      "patch_size": 16,     // 图像块大小
      "embed_dim": 512,     // 嵌入维度
      "num_heads": 16,      // 注意力头数
      "num_layers": 12,     // 层数
      "num_residual_blocks": 8  // 残差块数
    }
  }
}
```

#### 训练调用
```bash
python DLCSMRI/Transformer.py
```

#### 推理调用
在主程序中通过`TransformerInference`模块调用

### 模型4：ISTA-Net

#### 模型结构（通俗解释）
ISTA-Net将ISTA迭代过程展开为深度网络：
1. **迭代展开**：将传统迭代算法的每一步映射为网络层
2. **参数学习**：让网络自动学习最优的阈值和字典
3. **优势**：结合了优化理论和深度学习

#### 模型特点
- 理论指导设计
- 可解释性强
- 收敛性有保证

#### 配置参数
```json
{
  "ISTA": {
    "DEVICE": "cpu",
    "optim": {
      "num_iters": 8,       // 迭代次数
      "channels": 256,      // 通道数
      "rho_init": 0.1       // 初始参数
    }
  }
}
```

#### 训练调用
```bash
python DLCSMRI/ista.py
```

#### 推理调用
在主程序中通过`ISTAInference`模块调用

## 第三部分：数据处理工具详解

### 掩码生成工具（mask_2d.py）

#### 支持的采样模式
1. **相位编码采样**：在k空间中心区域强制采样，其余随机采样
2. **中心矩形采样**：中心区域矩形采样，其余随机采样
3. **径向采样**：沿径向方向的射线采样
4. **螺旋采样**：螺旋轨迹采样

#### 使用方法
在主程序的"数据处理"页面中选择相应的采样类型，设置参数后生成掩码文件。

### 数据集创建工具

#### 功能说明
自动创建标准的深度学习训练数据集结构：
```
dataset/
├── train/
│   ├── full/        # 完整采样图像
│   ├── undersampled/ # 欠采样k空间数据
│   └── mask/        # 采样掩码
└── val/
    ├── full/
    ├── undersampled/
    └── mask/
```

#### 使用方法
在"数据处理"页面点击"创建数据集结构"按钮。

## 第四部分：API调用说明（供开发人员使用）

### 压缩感知算法调用

#### ADMM-TV算法调用示例
```python
from Solver import ADMM_TV

# 设置参数
TV_params = {
    "tv_r": 5,
    "rho": 1,
    "n_iter": 50,
    "step": 0.5,
    "cg_iter": 3,
    "tv_ndim": 2
}

# 调用处理函数
results = ADMM_TV.process_kspace_files(
    kspace_paths,      # k空间文件路径列表
    mask_path,         # 掩码文件路径
    output_folder,     # 输出文件夹
    save_npy=True,     # 是否保存.npy文件
    TV_params=TV_params  # 算法参数
)
```

#### Wavelet-ISTA算法调用示例
```python
from Solver import Wavelet_ISTA

# 设置参数
wavelet_params = {
    "epsilon": 1e-4,
    "n_max": 50,
    "tol": 1e-4,
    "decfac": 0.5,
    "threshold": 5,
    "wavelet": 'haar',
    "level": 2
}

# 调用处理函数
results = Wavelet_ISTA.process_kspace_files(
    kspace_paths,
    mask_path,
    output_folder,
    save_npy=True,
    wavelet_params=wavelet_params
)
```

### 深度学习模型调用

#### U-Net模型训练调用
```python
from DLCSMRI.unet import MRIDenoisingTrainer

# 初始化训练器
trainer = MRIDenoisingTrainer(
    config_path="./DL_params.json",
    model_name="U-Net"
)

# 开始训练
trainer.train()
```

#### U-Net模型推理调用
```python
from DLCSMRI import UNetInference, unet
import torch

# 加载配置
config_loader = UNetInference.InferenceConfigLoader("./DL_params.json", "U-Net")
config = config_loader.config

# 初始化模型
model = unet.UNet(
    in_channels=config.get("IN_CHANNELS", 1),
    out_channels=config.get("OUT_CHANNELS", 1)
).to(config["DEVICE"])

# 加载预训练权重
model = UNetInference.load_model(model, config, "./model/U_Net.pth")

# 执行推理
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
```

## 第五部分：常见问题解答

### Q1: 如何选择合适的重建算法？
**A**: 
- 如果追求速度且对质量要求一般：选择ADMM-TV
- 如果需要高质量重建且时间充足：选择Wavelet-ISTA
- 如果有GPU且需要快速处理：选择ADMM-TV-CuPy
- 如果需要最高质量：选择深度学习方法

### Q2: 深度学习模型需要训练吗？
**A**: 
平台已提供预训练模型，可以直接使用。如果需要针对特定数据集优化效果，可以重新训练。

### Q3: 什么是k空间数据？
**A**: 
k空间是MRI数据的频域表示，包含了图像的所有信息。重建就是将k空间数据转换回图像域的过程。

### Q4: 什么是采样掩码？
**A**: 
采样掩码决定了在k空间中哪些位置被采样，哪些位置被忽略。它是实现压缩感知的关键。

### Q5: 如何准备输入数据？
**A**: 
输入数据需要是.npy格式的k空间数据和对应的掩码文件。可以通过"数据处理"工具生成。

## 第六部分：性能优化建议

### 算法参数调优
1. **迭代次数**：增加迭代次数通常能提高质量但会增加时间
2. **正则化参数**：根据图像特性调整，平衡保真度和正则化
3. **学习率**：深度学习中，学习率影响训练稳定性和收敛速度

### 硬件加速
1. **GPU加速**：使用CuPy版本算法可显著提升速度
2. **多核CPU**：算法支持多进程并行处理
3. **内存优化**：处理大图像时注意内存使用

## 第七部分：扩展开发指南

### 添加新算法
1. 在`Solver/`目录下创建新的算法实现文件
2. 实现标准的`process_kspace_files`函数接口
3. 在`MriRecon.py`中导入新模块
4. 在UI中添加相应的控件和信号槽连接

### 添加新深度学习模型
1. 在`DLCSMRI/`目录下创建模型训练和推理文件
2. 在`DL_params.json`中添加模型配置
3. 实现标准的数据加载和模型接口
4. 在主程序中添加模型选择选项

## 附录：技术术语解释

### MRI相关术语
- **k空间**：MRI数据的频域表示
- **傅里叶变换**：将信号在时域和频域之间转换的数学工具
- **欠采样**：只采集部分k空间数据以缩短扫描时间
- **信噪比**：信号强度与噪声强度的比值

### 算法相关术语
- **压缩感知**：用少量采样重建信号的理论
- **稀疏性**：信号在某种变换域中大部分系数为零或接近零
- **正则化**：在优化问题中加入约束条件防止过拟合
- **迭代算法**：通过重复计算逐步逼近最优解的方法

### 深度学习相关术语
- **神经网络**：模拟人脑神经元连接的计算模型
- **卷积**：提取图像特征的数学操作
- **注意力机制**：让模型关注重要信息的机制
- **迁移学习**：将在一个任务上学到的知识应用到另一个任务上