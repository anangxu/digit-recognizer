"""
LeNet5 模型定义与 MNIST 训练脚本
运行方式: python train.py
训练完成后会保存 lenet5_mnist.pth 权重文件，并输出训练曲线图 training_curves.png
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# LeNet-5 网络结构
# 输入: 1×28×28 (MNIST灰度图，原版32×32，此处适配28×28)
# Conv1(1→6, 5×5, pad=2) → AvgPool(2×2) → Conv2(6→16, 5×5) → AvgPool(2×2)
# → Flatten → FC(400→120) → FC(120→84) → FC(84→10)
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # 1×28×28 → 6×28×28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),        # 6×28×28 → 6×14×14
            nn.Conv2d(6, 16, kernel_size=5),              # 6×14×14 → 16×10×10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),        # 16×10×10 → 16×5×5
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


def get_dataloaders(data_dir='./data', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST 官方均值/标准差
    ])
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
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
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total * 100.0


def plot_curves(train_losses, test_losses, train_accs, test_accs, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss')
    axes[0].plot(epochs, test_losses, 'r-o', label='Test Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    axes[1].plot(epochs, test_accs, 'r-o', label='Test Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'训练曲线已保存至: {save_path}')


def main():
    # 超参数
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 1e-3
    DATA_DIR = './data'
    SAVE_PATH = 'lenet5_mnist.pth'
    CURVE_PATH = 'training_curves.png'

    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}', flush=True)

    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print(f'开始训练，共 {EPOCHS} 个 epoch...')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(
            f'Epoch [{epoch:02d}/{EPOCHS}] '
            f'Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  '
            f'Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%'
        )

    print('-' * 60)
    print(f'最终测试精度: {test_accs[-1]:.2f}%')

    torch.save(model.state_dict(), SAVE_PATH)
    print(f'模型权重已保存至: {SAVE_PATH}')

    plot_curves(train_losses, test_losses, train_accs, test_accs, CURVE_PATH)


if __name__ == '__main__':
    main()
