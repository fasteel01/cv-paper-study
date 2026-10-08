"""
GoogLeNet 학습 스크립트 (데모용).

논문은 ImageNet으로 학습하지만, 여기서는 파이프라인 확인 목적으로 CIFAR-10을 64x64로 리사이즈해 사용한다.
(모델이 AdaptiveAvgPool을 쓰므로 해상도가 달라도 동작한다.)
학습 시 총 손실 = main + 0.3 * aux1 + 0.3 * aux2 (논문과 동일), 평가는 main 출력만 사용.

실행:
    pip install torch torchvision
    python train.py --epochs 3
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import GoogLeNet

AUX_WEIGHT = 0.3


def get_dataloaders(batch_size: int, size: int):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.batch_size, args.img_size)

    model = GoogLeNet(num_classes=args.num_classes, aux_logits=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            main_out, aux1, aux2 = model(x)
            loss = criterion(main_out, y) + AUX_WEIGHT * criterion(aux1, y) + AUX_WEIGHT * criterion(aux2, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f} test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
