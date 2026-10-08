"""
BatchNorm 적용 전/후 비교 실험 (CIFAR-10).

같은 구조의 SmallCNN을 BN 유무 x 학습률 조합으로 학습해 에폭별 test accuracy를 비교한다.
기대 현상: BN이 있으면 더 빨리 수렴하고, 큰 학습률(예: 0.1)에서도 안정적이다.
BN이 없으면 큰 학습률에서 발산하거나 학습이 매우 느릴 수 있다.

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

from model import SmallCNN


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
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2),
    )


def evaluate(model, loader, device):
    model.eval()  # BN은 eval에서 running 통계를 사용
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(use_bn, lr, epochs, train_loader, test_loader, device):
    torch.manual_seed(0)
    model = SmallCNN(use_bn=use_bn).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"[bn={use_bn} lr={lr}] epoch {epoch}/{epochs} loss={running / n:.4f} test_acc={acc:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.1])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    results = {}
    for lr in args.lrs:
        for use_bn in (False, True):
            results[(use_bn, lr)] = run(use_bn, lr, args.epochs, train_loader, test_loader, device)

    print("\n=== 요약: 마지막 에폭 test accuracy ===")
    for (use_bn, lr), hist in results.items():
        print(f"BN={use_bn!s:5} lr={lr:<5} final_acc={hist[-1]:.4f}")


if __name__ == "__main__":
    main()
