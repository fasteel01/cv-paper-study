"""
ResNet vs Plain 비교 실험 (CIFAR-10).

같은 깊이의 망에서 shortcut 유무(residual=True/False)만 바꿔 학습해 degradation 현상을 확인한다.
깊을수록(--arch resnet34 등) plain 망의 학습 손실이 더 잘 안 떨어지는 경향을 볼 수 있다.

실행:
    pip install torch torchvision
    python train.py --arch resnet18 --epochs 10
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model as M


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
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(arch, residual, epochs, lr, train_loader, test_loader, device):
    torch.manual_seed(0)
    model = getattr(M, arch)(num_classes=10, cifar_stem=True, residual=residual).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    final = None
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
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        final = (running / n, acc)
        print(f"[{arch} residual={residual}] epoch {epoch}/{epochs} train_loss={final[0]:.4f} test_acc={acc:.4f}")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    results = {}
    for residual in (False, True):
        results[residual] = run(args.arch, residual, args.epochs, args.lr, train_loader, test_loader, device)

    print("\n=== 요약 (마지막 에폭) ===")
    for residual, (loss, acc) in results.items():
        print(f"{args.arch} {'ResNet(skip)' if residual else 'Plain':13} train_loss={loss:.4f} test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
