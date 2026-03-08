# 手写数字识别 App

基于 PyTorch 训练 LeNet-5 模型，部署到 Android 端实现实时手写数字识别。

## 项目结构

```
├── train/              # Python 训练代码
│   ├── train.py        # LeNet-5 定义 + MNIST 训练
│   ├── export.py       # 导出 TorchScript (.ptl)
│   └── requirements.txt
├── android/            # Android Studio 工程
│   └── app/src/main/
│       ├── java/com/example/digitrecognizer/
│       │   ├── MainActivity.kt
│       │   ├── DigitClassifier.kt
│       │   └── ImagePreprocessor.kt
│       └── res/
└── report/
    └── report.md       # 项目报告
```

## 快速开始

### 1. 训练模型

```bash
cd train
pip install -r requirements.txt
python train.py        # 训练，生成 lenet5_mnist.pth（约15分钟）
python export.py       # 导出，自动复制到 android/app/src/main/assets/
```

### 2. 运行 Android App

1. 用 Android Studio 打开 `android/` 目录
2. 等待 Gradle 同步完成
3. 确认 `android/app/src/main/assets/digit_model.ptl` 存在
4. 连接设备或启动模拟器，点击 Run

## 技术栈

| 模块 | 技术 |
|------|------|
| 模型训练 | Python 3.9+, PyTorch, torchvision |
| 模型部署 | TorchScript Lite (`torch.jit.trace`) |
| Android | Kotlin, CameraX 1.3.1, PyTorch Mobile 1.13.1 |

## 模型性能

- MNIST 测试集精度：**98.94%**
- 模型大小：~255 KB
- 推理延迟：~300ms（模拟器）

## 图像预处理 Pipeline

```
摄像头图像 → 灰度化 → 自适应局部阈值二值化
→ BFS 最大连通区域（轮廓检测）→ Bounding Box 裁剪
→ Resize 28×28 → 黑底白字（与 MNIST 一致）→ 归一化 → 模型推理
```

## 注意事项

- 训练数据（MNIST）和模型权重不随代码上传，运行 `train.py` 自动下载
- `digit_model.ptl` 需要运行 `export.py` 后手动放入 `android/app/src/main/assets/`
- 建议使用白纸黑字、光线充足的环境进行识别
