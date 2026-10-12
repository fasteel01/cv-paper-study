"""
MobileNet 비교 학습 스크립트 (데모용).

CIFAR-10에서 depthwise separable conv 버전과 표준 conv 버전을 같은 설정으로 학습해
파라미터 수와 정확도를 비교한다 (논문 Table 4의 축소판).
32x32 입력이므로 첫 conv의 stride를 1로 둔다 (--first-stride).

실행:
    pip install torch torchvision
    python train.py --epochs 10
    python train.py --epochs 10 --width-mult 0.5
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MobileNetV1, count_params

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


def run(separable: bool, args, train_loader, test_loader, device):
    tag = "separable" if separable else "standard "
    model = MobileNetV1(
        num_classes=10, width_mult=args.width_mult, separable=separable, first_stride=args.first_stride
    ).to(device)
    print(f"\n=== {tag} conv | params={count_params(model) / 1e6:.2f}M ===")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    acc = 0.0
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
        acc = evaluate(model, test_loader, device)
        print(f"[{tag}] epoch {epoch}/{args.epochs} loss={running_loss / len(train_loader.dataset):.4f} test_acc={acc:.4f}")
    return count_params(model), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--first-stride", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    results = {}
    for separable in (True, False):
        results[separable] = run(separable, args, train_loader, test_loader, device)

    print("\n--- 요약 ---")
    for separable, (n, acc) in results.items():
        print(f"{'separable' if separable else 'standard '}: params={n / 1e6:.2f}M test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
