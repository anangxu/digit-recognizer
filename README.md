# 手写数字识别 App

基于 PyTorch 训练带残差连接的 DigitCNN 模型，部署到 Android 端实现实时手写数字识别。支持置信度条形图、扫描线动画、深色现代主题。

## 项目结构

```
├── train/
│   ├── train.py            # DigitCNN 定义 + MNIST 训练（含数据增强）
│   ├── export.py           # 导出 TorchScript (.ptl) 并复制到 Android assets
│   └── requirements.txt
├── android/
│   └── app/src/main/
│       ├── java/com/example/digitrecognizer/
│       │   ├── MainActivity.kt          # 相机、推理、UI 逻辑
│       │   ├── DigitClassifier.kt       # PyTorch Mobile 模型推理封装
│       │   ├── ImagePreprocessor.kt     # 图像预处理 pipeline
│       │   └── ConfidenceBarView.kt     # 自定义置信度条形图 View
│       └── res/
│           ├── anim/                    # digit_enter.xml, scan_line.xml
│           ├── drawable/                # gradient_top.xml, viewfinder_border.xml
│           └── layout/activity_main.xml
└── report/
    └── report.md
```

## 快速开始

### 1. 训练模型

```bash
cd train
pip install -r requirements.txt
python train.py        # 训练 30 epoch，生成 lenet5_mnist.pth（CPU 约 1 小时）
python export.py       # 导出，自动复制到 android/app/src/main/assets/
```

### 2. 构建 Android App

```bash
cd android
# Windows
gradlew.bat assembleDebug
# 生成路径：app/build/outputs/apk/debug/app-debug.apk
```

> **注意**：需要 JDK 17（推荐使用 Android Studio 自带的 jbr-17）。  
> 环境变量设置：`JAVA_HOME=C:\Users\<用户名>\.jdks\jbr-17.x.x`

或直接用 Android Studio 打开 `android/` 目录，点击 Run。

## 技术栈

| 模块 | 技术 |
|------|------|
| 模型训练 | Python 3.9+, PyTorch, torchvision |
| 网络结构 | DigitCNN（带残差块，1.15M 参数） |
| 数据增强 | 随机旋转/透视/仿射/模糊/光照变化 |
| 模型部署 | TorchScript Lite (`_save_for_lite_interpreter`) |
| Android | Kotlin, CameraX 1.3.1, PyTorch Mobile Lite 2.1.0 |
| 构建环境 | AGP 7.4.2, Gradle 7.6.1, JDK 17 |

## 模型性能

- MNIST 测试集精度：**≥ 99.4%**（DigitCNN + 数据增强，30 epoch）
- 模型大小：~4 MB
- 推理延迟：~300ms（模拟器）

## 图像预处理 Pipeline

```
摄像头帧
  → 裁剪取景框
  → 灰度化
  → 高斯模糊（去噪）
  → Otsu 全局阈值（失败时降级为自适应局部阈值）
  → 颜色极性自动纠正（兼容白底黑字/黑底白字）
  → 形态学闭运算（连接断笔画）
  → 连通域分析 + 合并 BBox
  → 正方形裁剪（25% padding）
  → 缩放到 20×20（NEAREST，保留笔画锐度）
  → 居中放置到 28×28 黑色画布（模拟 MNIST 留白）
  → MNIST 归一化 (mean=0.1307, std=0.3081)
  → 模型推理 → Softmax → 置信度分布
```

## UI 功能

- **实时识别**：CameraX 持续帧分析，每秒多次推理
- **置信度条形图**：实时显示 0–9 每个数字的概率分布
- **扫描线动画**：取景框内上下扫描动效
- **数字切换动画**：识别结果变化时滑入弹动效果
- **拍照冻结**：点击按钮冻结当前帧，便于查看识别结果

## 注意事项

- 训练数据（MNIST）和模型权重不随代码上传，运行 `train.py` 自动下载
- `digit_model.ptl` 需要运行 `export.py` 后才会出现在 `android/app/src/main/assets/`
- 建议在光线充足、背景干净的环境使用
- 支持白纸黑字和黑板白字两种场景（自动极性检测）
