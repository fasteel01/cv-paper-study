"""
DenseNet CIFAR-10 학습 스크립트 (데모용).

논문 설정(300 epoch, SGD+Nesterov, lr 0.1 step decay)을 축약해 cosine 스케줄로 짧게 돌린다.

실행:
    pip install torch torchvision
    python train.py --epochs 10
    python train.py --depth 40 --growth-rate 12 --no-bc --epochs 10
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import count_params, densenet_cifar

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_dataloaders(batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--growth-rate", type=int, default=12)
    parser.add_argument("--no-bc", action="store_true", help="bottleneck/compression 중 compression을 끈 비-BC 버전")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = densenet_cifar(args.depth, args.growth_rate, bc=not args.no_bc).to(device)
    print(f"DenseNet depth={args.depth} k={args.growth_rate} bc={not args.no_bc} params={count_params(model) / 1e6:.2f}M")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f} test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
