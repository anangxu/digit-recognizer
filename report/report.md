# 手写数字识别 App 项目报告

**项目名称**：基于 PyTorch + Android 的实时手写数字识别  
**完成日期**：2026 年 3 月  

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [系统架构](#2-系统架构)
3. [网络结构：DigitCNN](#3-网络结构digitcnn)
4. [模型训练过程](#4-模型训练过程)
5. [模型导出与移动端部署](#5-模型导出与移动端部署)
6. [Android 端实现](#6-android-端实现)
7. [图像预处理流程](#7-图像预处理流程)
8. [UI 设计](#8-ui-设计)
9. [实际效果展示](#9-实际效果展示)
10. [总结与不足](#10-总结与不足)

---

## 1. 项目背景与目标

手写数字识别是计算机视觉领域的经典任务，也是深度学习工程落地的标准案例。本项目完整实现从**模型训练**到**移动端实时部署**的全流程，具体目标：

- 使用 PyTorch 训练带残差连接的卷积网络，在 MNIST 数据集上达到 ≥ 99.4% 测试精度
- 通过大量数据增强缩小训练域与真实拍摄域之间的差距（Domain Gap）
- 将模型通过 TorchScript 转换并部署到 Android 端，实现完全离线推理
- 实现摄像头实时取景、手写数字识别、0-9 置信度分布展示的完整交互

---

## 2. 系统架构

```
PC 端（Python）                              Android 端（Kotlin）
─────────────────────────                   ──────────────────────────────────
MNIST 数据集（含数据增强）                    CameraX 实时预览
        ↓                                              ↓
  DigitCNN 训练（PyTorch）    ──模型──▶       ImagePreprocessor
        ↓                   digit_model.ptl   高斯去噪→Otsu二值化→形态学→28×28
  导出 TorchScript (.ptl)                              ↓
                                              DigitClassifier（PyTorch Mobile Lite）
                                                        ↓
                                              UI：识别结果 + 0-9 置信度条形图
```

**技术栈：**

| 模块 | 技术 |
|------|------|
| 模型训练 | Python 3.14, PyTorch 2.x, torchvision |
| 网络结构 | DigitCNN（带残差块，1.15M 参数） |
| 数据增强 | 旋转/透视/仿射/高斯模糊/光照变化/随机擦除 |
| 模型导出 | `torch.jit.trace` + `optimize_for_mobile` |
| Android 开发语言 | Kotlin |
| 推理引擎 | PyTorch Mobile Lite 2.1.0 |
| 摄像头框架 | CameraX 1.3.1 |
| UI 组件 | Material Components, ConstraintLayout, 自定义 View |
| 构建环境 | AGP 7.4.2, Gradle 7.6.1, JDK 17 |

---

## 3. 网络结构：DigitCNN

### 3.1 设计思路

LeNet-5（1998年，6万参数）在 MNIST 干净数据上效果好，但面对真实拍照的光照变化、透视扭曲等情况泛化能力不足。本项目升级为带**残差连接**的现代卷积网络 DigitCNN，参数量约 **115 万**，同时通过 BatchNorm 和 Dropout 提升训练稳定性和泛化能力。

### 3.2 网络层参数

| 阶段 | 结构 | 输入尺寸 | 输出尺寸 |
|------|------|---------|---------|
| Stem | Conv(1→32, 3×3) + BN + ReLU | 1×28×28 | 32×28×28 |
| Stage1 | ResBlock(32) + Conv(32→64, 3×3, stride=2) + BN + ReLU | 32×28×28 | 64×14×14 |
| Stage2 | ResBlock(64) + Conv(64→128, 3×3, stride=2) + BN + ReLU | 64×14×14 | 128×7×7 |
| Stage3 | ResBlock(128) + Conv(128→128, 3×3, stride=2) + BN + ReLU | 128×7×7 | 128×4×4 |
| Head | Flatten + FC(2048→256) + ReLU + Dropout(0.4) + FC(256→10) | 128×4×4 | 10 |

**ResBlock 结构：**
```
输入 x
  → Conv(3×3) + BN + ReLU
  → Conv(3×3) + BN
  → + x（残差相加）
  → ReLU
输出
```

**总参数量**：约 **1,155,690** 个，模型导出后约 **4 MB**。

### 3.3 与 LeNet-5 对比

| 指标 | LeNet-5 | DigitCNN（本项目） |
|------|---------|-----------------|
| 参数量 | ~6.2 万 | ~115.6 万 |
| 激活函数 | Tanh | ReLU |
| 归一化 | 无 | BatchNorm |
| 正则化 | 无 | Dropout(0.4) |
| 残差连接 | 无 | 有 |
| MNIST 测试精度 | ~98.9% | **≥ 99.5%** |

---

## 4. 模型训练过程

### 4.1 数据集

- **MNIST**：60,000 训练样本 + 10,000 测试样本，28×28 灰度手写数字 0-9

### 4.2 数据增强策略

训练集加入了针对真实拍照场景的增强，模拟手机拍照时常见的各种干扰：

| 增强方法 | 参数 | 模拟场景 |
|---------|------|---------|
| RandomRotation | ±20° | 书写倾斜 |
| RandomAffine | 平移10%, 缩放80-120%, 剪切15° | 视角偏移 |
| RandomPerspective | distortion=0.3, p=0.5 | 手机拍摄角度 |
| GaussianBlur | σ∈[0.1, 1.5], p=0.3 | 对焦不准/运动模糊 |
| ColorJitter | brightness=0.4, contrast=0.4, p=0.5 | 光照不均 |
| RandomErasing | scale=[0.02, 0.1], p=0.1 | 部分遮挡 |

### 4.3 训练配置

| 超参数 | 值 |
|--------|-----|
| Batch Size | 128 |
| Epochs | 30 |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | CosineAnnealingLR (T_max=30, η_min=1e-5) |
| 损失函数 | CrossEntropyLoss (label_smoothing=0.1) |
| 混合精度 | 自动（有 GPU 时开启） |

### 4.4 训练结果

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|-----------|----------|
| 01 | 0.7921 | 90.10% | 0.5558 | 98.73% |
| 03 | 0.6069 | 97.38% | 0.5288 | 99.37% |
| 05 | 0.5879 | 97.94% | 0.5233 | 99.43% |
| 07 | 0.5776 | 98.18% | 0.5214 | **99.54%** ★ |
| 09 | 0.5695 | 98.40% | 0.5173 | **99.59%** ★ |
| 10 | 0.5675 | 98.40% | 0.5167 | 99.58% |
| 30 | 训练中... | — | — | 预计 ≥ 99.6% |

> ★ 表示当时最佳测试精度，自动保存权重。训练曲线见 `train/training_curves.png`

### 4.5 关键设计决策

**Label Smoothing（标签平滑）**：将硬标签（0/1）软化为 0.9/0.1，相当于正则化，防止模型对预测过于自信，提升泛化性。

**CosineAnnealingLR**：学习率从 1e-3 按余弦曲线衰减到 1e-5，使训练后期在最优点附近细粒度收敛，比 StepLR 效果更稳定。

**AdamW + weight_decay**：相比 Adam，AdamW 将 L2 正则化从梯度更新中解耦，效果更正确，配合 Dropout 共同抑制过拟合。

---

## 5. 模型导出与移动端部署

### 5.1 导出流程

```python
# 加载最佳权重
model = DigitCNN()
model.load_state_dict(torch.load('lenet5_mnist.pth'))
model.eval()

# TorchScript trace（静态图，不依赖 Python 运行时）
example = torch.zeros(1, 1, 28, 28)
traced = torch.jit.trace(model, example)

# 移动端优化（算子融合、量化友好转换）
optimized = optimize_for_mobile(traced)
optimized._save_for_lite_interpreter('digit_model.ptl')
```

### 5.2 为什么使用 TorchScript

| 方案 | 依赖 Python | Android 可用 | 大小 |
|------|------------|-------------|------|
| .pth 权重 | 是 | 否 | ~4.4 MB |
| TorchScript .ptl | 否 | 是 | ~4 MB |

TorchScript 将模型序列化为独立的中间表示，可在 C++/Java/Kotlin 中直接推理，`optimize_for_mobile` 进一步做算子融合，减少推理延迟。

### 5.3 部署步骤

```bash
cd train
python train.py     # 生成 lenet5_mnist.pth（最佳 epoch 自动保存）
python export.py    # 生成 digit_model.ptl，自动复制到 android/app/src/main/assets/

cd ../android
gradlew.bat assembleDebug   # 生成 app-debug.apk（~80 MB，含 PyTorch 原生库）
```

---

## 6. Android 端实现

### 6.1 模块职责

```
MainActivity.kt         — 权限管理、CameraX 初始化、每300ms推理、UI更新、动画控制
DigitClassifier.kt      — 加载 .ptl 模型、Bitmap→Tensor、推理、Softmax、返回10类概率
ImagePreprocessor.kt    — 完整图像预处理 pipeline（见第7节）
ConfidenceBarView.kt    — 自定义 View，动态绘制 0-9 置信度水平条形图
```

### 6.2 推理时序

```
CameraX ImageAnalysis（连续帧）
        ↓ 每300ms 触发一次
  getViewfinderRect() → 屏幕中心取景框区域
        ↓
  ImagePreprocessor.process(bitmap, rect) → 28×28 黑底白字 Bitmap
        ↓
  DigitClassifier.classify(bitmap28x28)
    → bitmapToTensor()：像素归一化 (x - 0.1307) / 0.3081
    → module.forward(IValue.from(tensor))
    → softmax(logits) → 10类概率
        ↓
  updateResultUI(result)
    → 数字变化时触发 digit_enter 动画
    → ConfidenceBarView.setScores(allScores, predictedDigit)
```

### 6.3 UI 交互状态

| 状态 | 说明 |
|------|------|
| 实时模式 | 每300ms识别，取景框扫描线持续动画 |
| 拍照确认 | 冻结当前帧，扫描线停止 |
| 重新识别 | 解冻，恢复实时识别和扫描线 |
| 模型加载中 | 启动时异步加载，完成后隐藏遮罩 |
| 权限拒绝 | 显示授权按钮 |

---

## 7. 图像预处理流程

真实摄像头环境与 MNIST 存在显著差异（光照、背景、透视等），需要完整的预处理 pipeline 将真实图像"拉回"到接近 MNIST 的分布。

```
摄像头帧
  ↓ 1. 裁剪取景框区域
  ↓ 2. 灰度化（ColorMatrix.setSaturation=0）
  ↓ 3. 高斯模糊（3×3，σ≈1.0）去除摄像头噪点
  ↓ 4. Otsu 全局阈值二值化
       → 若前景比例 < 1% 或 > 60%（光照极端情况），
         降级为自适应局部阈值（积分图加速，blockSize=21, C=10）
  ↓ 5. 颜色极性自动纠正
       → 比较前景/背景平均灰度，若前景更亮（极性反转）则翻转
       → 保证始终输出"黑底白字"
  ↓ 6. 形态学闭运算（膨胀→腐蚀，radius=1）
       → 连接断开的笔画，填补笔画内空洞
  ↓ 7. 连通域分析（BFS 4邻域）
       → 过滤噪点（面积 < 0.2% 或 > 85%）
       → 合并所有有效连通域的 BBox（兼容断笔）
  ↓ 8. 正方形裁剪（BBox + 25% padding）
  ↓ 9. 缩放至 20×20（NEAREST 插值，保留二值锐度）
  ↓ 10. 居中放置到 28×28 黑色画布（四周各 4px 留白，模拟 MNIST 分布）
  ↓
  输出：28×28 黑底白字 Bitmap → DigitClassifier 归一化推理
```

### 关键设计决策

| 问题 | 解决方案 | 原因 |
|------|---------|------|
| 摄像头噪点 | 高斯模糊预处理 | 避免噪点被误识别为前景 |
| 光照不均 | Otsu+自适应双策略 | 单一策略在极端光照下失败 |
| 黑板/暗背景场景 | 颜色极性自动检测 | MNIST 是黑底白字，需自动对齐 |
| 断笔画 | 形态学闭运算 | 连接 "8" "0" 等数字的间断笔画 |
| 数字位置偏移 | 连通域合并BBox | 自动定位，无需手动对准 |
| 缩放模糊 | NEAREST 插值 | 保留二值图的锐利边缘 |
| MNIST留白分布 | 20→28 居中嵌入 | 训练数据四周均有留白，保持一致 |

---

## 8. UI 设计

### 8.1 整体风格

采用现代深色主题，主色调为霓虹青（`#00E5FF`）和深背景（`#090E1A`），视觉风格接近专业 AI 应用。

### 8.2 主要 UI 元素

**取景框扫描线动画**（`scan_line.xml`）
- 一条青色半透明横线在取景框内上下循环扫描
- 拍照冻结时自动停止，恢复实时识别时重新启动

**识别数字切换动画**（`digit_enter.xml`）
- 当识别结果变化时，新数字从上方滑入并带有弹性回弹效果
- 使用 `overshoot_interpolator` 实现弹动感

**置信度条形图**（`ConfidenceBarView.kt`）
- 自定义 View，横向排列 0-9 共 10 个进度条
- 当前预测数字用霓虹渐变色高亮，其余数字灰色
- 每次更新时动画平滑过渡到新值

**顶部渐变遮罩**（`gradient_top.xml`）
- 顶部半透明深色渐变，提升文字可读性

### 8.3 布局结构

```
ConstraintLayout（全屏）
  ├── PreviewView（相机预览，全屏）
  ├── gradient_top（顶部渐变遮罩）
  ├── scanLine（扫描线，约束在取景框内）
  ├── viewfinderBorder（取景框边框）
  ├── tvHint（提示文字）
  ├── cardResult（结果卡片）
  │     ├── tvDigit（识别数字，大字）
  │     ├── divider（分隔线）
  │     └── confidenceBarView（0-9置信度条形图）
  └── btnCapture（拍照/重识别按钮）
```

---

## 9. 实际效果展示

### 9.1 测试环境

- 设备：Android 模拟器（Pixel 6, API 33）
- 摄像头：笔记本内置摄像头对准手写纸张

### 9.2 测试结果

| 测试场景 | 结果 |
|---------|------|
| 白纸黑字、光线充足 | 准确率高，响应 ~300ms |
| 字迹适中粗细 | 各数字均可正确识别 |
| 轻微角度倾斜 | 数据增强训练后明显改善 |
| 黑底白字（颜色反转） | 自动极性检测后正确识别 |

### 9.3 推荐演示流程（录屏约 1 分钟）

| 时间 | 内容 |
|------|------|
| 0-5s | App 启动，模型加载 |
| 5-25s | 依次识别 1、3、7、2（形状各异，效果好）|
| 25-35s | 点击"拍照确认"，展示冻结 + 置信度条形图 |
| 35-45s | 点击"重新识别"，恢复扫描线动画 |
| 45-60s | 展示数字切换时的滑入弹动动画效果 |

---

## 10. 总结与不足

### 10.1 项目成果

| 指标 | 结果 |
|------|------|
| MNIST 测试精度 | **≥ 99.59%**（训练中，持续提升） |
| 模型大小 | ~4 MB（含残差结构） |
| Android 推理响应 | ~300ms/帧 |
| 部署方式 | PyTorch Mobile Lite，完全离线 |
| 代码仓库 | https://github.com/anangxu/digit-recognizer |

完整实现了从数据增强训练到移动端实时部署的全流程，掌握了残差网络设计、TorchScript 模型导出、CameraX 开发、图像处理算法（Otsu/自适应阈值/形态学/连通域分析）等核心技术。

### 10.2 主要不足与改进方向

| 不足 | 原因 | 改进方向 |
|------|------|---------|
| 拍摄角度偏大时识别率下降 | 无透视矫正 | Hough 变换检测纸张四角，做投影变换 |
| 笔画极细/极粗时效果差 | MNIST 笔画宽度相对固定 | 预处理中加入笔画宽度归一化 |
| 复杂背景（桌面纹理）干扰 | 连通域合并可能引入背景噪点 | 面积过滤阈值自适应调整 |
| 只支持单个数字 | 模型设计限制 | 结合目标检测实现多数字序列识别 |

### 10.3 技术心得

本项目最大的挑战在于**训练域与部署域的差距（Domain Gap）**：模型在 MNIST 上接近完美，但真实摄像头图像与训练数据差异显著。解决这一问题需要两手并进——在训练侧用数据增强覆盖真实场景，在推理侧用预处理 pipeline 把真实图像变换回接近 MNIST 的分布。这也是深度学习工程落地时面临的核心挑战之一。

---

*报告完*
