"""
模型导出脚本：将训练好的 DigitCNN 权重导出为 TorchScript (.ptl) 格式
运行方式: python export.py
前提: 已运行 train.py 并生成 lenet5_mnist.pth
输出: digit_model.ptl（同时复制到 Android assets 目录）
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


# 必须与 train.py 中保持完全一致的网络结构
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
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
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f'已加载权重: {weights_path}')
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    example_input = torch.zeros(1, 1, 28, 28)
    traced = torch.jit.trace(model, example_input)

    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(output_path)
    print(f'模型已导出至: {output_path}')

    if os.path.isdir(android_assets_dir):
        dest = os.path.join(android_assets_dir, 'digit_model.ptl')
        shutil.copy2(output_path, dest)
        print(f'已自动复制到 Android assets: {dest}')
    else:
        print(f'提示: 请手动将 {output_path} 复制到 Android 工程的 app/src/main/assets/')

    loaded = torch.jit.load(output_path)
    loaded.eval()
    with torch.no_grad():
        out = loaded(example_input)
    print(f'导出模型验证通过，输出形状: {out.shape}')


if __name__ == '__main__':
    export_model()
