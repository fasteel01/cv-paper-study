"""
VGG-11 학습 스크립트 (데모용): BatchNorm 없음 vs 있음 비교.

논문은 ImageNet으로 학습하지만, 여기서는 파이프라인 확인 목적으로 CIFAR-10(32x32)을 사용한다.
작은 입력에 맞춰 분류기를 (pool_size=1, hidden_dim=512)로 줄였다.
깊은 plain 망(VGG)이 BN 유무에 따라 학습이 얼마나 달라지는지 관찰해 보자.

실행:
    pip install torch torchvision
    python train.py --epochs 5
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VGG


def get_dataloaders(batch_size: int):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(batch_norm: bool, args, device, train_loader, test_loader):
    tag = "with BN" if batch_norm else "no BN"
    model = VGG("A", num_classes=10, batch_norm=batch_norm, pool_size=1, hidden_dim=512).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"[{tag}] epoch {epoch}/{args.epochs} loss={train_loss:.4f} test_acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size)
    for bn in (False, True):
        run(bn, args, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
