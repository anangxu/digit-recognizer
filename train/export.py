"""
模型导出脚本：将训练好的 LeNet5 权重导出为 TorchScript (.ptl) 格式
运行方式: python export.py
前提: 已运行 train.py 并生成 lenet5_mnist.pth
输出: digit_model.ptl（同时复制到 Android assets 目录，如果路径存在）
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def export_model(
    weights_path: str = 'lenet5_mnist.pth',
    output_path: str = 'digit_model.ptl',
    android_assets_dir: str = '../android/app/src/main/assets',
):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'权重文件 {weights_path} 不存在，请先运行 train.py'
        )

    device = torch.device('cpu')
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f'已加载权重: {weights_path}')

    # 使用 trace 转换为 TorchScript（输入为标准 MNIST 格式: 1×1×28×28）
    example_input = torch.zeros(1, 1, 28, 28)
    traced = torch.jit.trace(model, example_input)

    # 针对移动端优化（减小模型体积、提升推理速度）
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(output_path)
    print(f'模型已导出至: {output_path}')

    # 如果 Android assets 目录存在，自动复制
    if os.path.isdir(android_assets_dir):
        dest = os.path.join(android_assets_dir, 'digit_model.ptl')
        shutil.copy2(output_path, dest)
        print(f'已自动复制到 Android assets: {dest}')
    else:
        print(f'提示: Android assets 目录 {android_assets_dir} 不存在，')
        print(f'     请手动将 {output_path} 复制到 Android 工程的 app/src/main/assets/ 目录下')

    # 验证导出的模型可以正常推理
    loaded = torch.jit.load(output_path)
    loaded.eval()
    with torch.no_grad():
        out = loaded(example_input)
    print(f'导出模型验证通过，输出形状: {out.shape}，预测类别: {out.argmax().item()}')


if __name__ == '__main__':
    export_model()
