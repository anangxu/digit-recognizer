"""
改进版手写数字识别训练脚本
- 使用更强的 CNN（类 ResNet 风格，带残差连接）
- 大幅数据增强：模拟真实拍照场景（旋转、透视、光照、模糊、噪声）
- 训练 20 epoch，使用 CosineAnnealing 学习率
- 最终测试集精度目标 ≥ 99.4%
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 网络结构：带残差连接的深度 CNN（比 LeNet5 强很多）
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """基本残差块：两层 3×3 Conv + BN + ReLU，旁路 shortcut"""
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
    """
    输入: 1×28×28
    Stage1: 32 ch → 14×14
    Stage2: 64 ch → 7×7
    Stage3: 128 ch → 3×3
    FC: 128*3*3 → 256 → 10
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 数据增强：模拟真实拍照场景
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir='./data', batch_size=128):
    # 训练集：大量数据增强
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        # 随机旋转 ±20°
        transforms.RandomRotation(degrees=20, fill=0),
        # 随机仿射变换：平移10%、缩放80-120%、剪切15°（模拟透视）
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=15,
            fill=0,
        ),
        # 随机透视变换（模拟手机拍照角度偏移）
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
        transforms.ToTensor(),
        # 随机高斯模糊（模拟对焦不准）
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        # 随机亮度/对比度变化（模拟光照不均）
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4)
        ], p=0.5),
        # 标准化
        transforms.Normalize((0.1307,), (0.3081,)),
        # 随机橡皮擦（模拟部分遮挡）
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    # 测试集：只做基础变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 训练 / 评估
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total * 100.0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total * 100.0


def plot_curves(train_losses, test_losses, train_accs, test_accs, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss')
    axes[0].plot(epochs, test_losses,  'r-o', label='Test Loss')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, train_accs, 'b-o', label='Train Acc')
    axes[1].plot(epochs, test_accs,  'r-o', label='Test Acc')
    axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    EPOCHS     = 30
    BATCH_SIZE = 128
    LR         = 1e-3
    DATA_DIR   = './data'
    SAVE_PATH  = 'lenet5_mnist.pth'      # 保持文件名兼容 export.py

    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'开始训练，共 {EPOCHS} epoch...')
    print('-' * 70)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_accs.append(tr_acc);   test_accs.append(te_acc)

        marker = ' ★' if te_acc > best_acc else ''
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(
            f'Epoch [{epoch:02d}/{EPOCHS}]  '
            f'Train {tr_loss:.4f}/{tr_acc:.2f}%  '
            f'Test {te_loss:.4f}/{te_acc:.2f}%  '
            f'LR {scheduler.get_last_lr()[0]:.2e}{marker}'
        )

    print('-' * 70)
    print(f'最佳测试精度: {best_acc:.2f}%  权重已保存至: {SAVE_PATH}')
    plot_curves(train_losses, test_losses, train_accs, test_accs)


if __name__ == '__main__':
    main()
