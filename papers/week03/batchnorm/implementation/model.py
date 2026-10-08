"""
Batch Normalization (Ioffe & Szegedy, 2015) — from-scratch PyTorch implementation.

구성:
  - BatchNorm2d : 학습 시 미니배치 통계(채널별, 배치 x H x W 기준), 추론 시 running 통계 사용
  - SmallCNN    : BN 적용 전/후 비교용 작은 CNN (Conv -> BN -> ReLU 순서, 논문 방식)

원 논문과의 차이점(현대적 단순화):
  - 원 논문은 Inception 기반 ImageNet 모델에 적용했지만, 여기서는 비교 실험용 작은 CNN을 사용한다.
  - running 분산은 PyTorch 관례대로 불편 분산(m/(m-1))으로 갱신하고, 정규화에는 편향 분산을 쓴다.
  - momentum은 PyTorch 정의(새 통계의 가중치, 기본 0.1)를 따른다.
"""

import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    """입력: (N, C, H, W). 채널별로 (N, H, W)에 대해 정규화한 뒤 gamma * x_hat + beta."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)  # 정규화에는 편향 분산
            with torch.no_grad():
                m = x.numel() / x.size(1)  # 채널당 원소 수 N*H*W
                unbiased_var = var * m / max(m - 1, 1)
                self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(unbiased_var * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]


class SmallCNN(nn.Module):
    """CIFAR-10(3x32x32)용 작은 CNN. use_bn=True면 Conv 뒤(ReLU 앞)에 BatchNorm 삽입."""

    def __init__(self, num_classes: int = 10, use_bn: bool = True, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            for _ in range(2):
                # BN이 있으면 bias는 beta가 대신하므로 불필요
                layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        return self.classifier(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) torch.nn.BatchNorm2d와 출력 비교 (train / eval)
    x = torch.randn(8, 16, 10, 10) * 3 + 5
    mine, ref = BatchNorm2d(16), nn.BatchNorm2d(16)
    mine.train()
    ref.train()
    out_m, out_r = mine(x), ref(x)
    print("train max diff:", (out_m - out_r).abs().max().item())
    print("running_mean diff:", (mine.running_mean - ref.running_mean).abs().max().item())
    print("running_var  diff:", (mine.running_var - ref.running_var).abs().max().item())
    print("normalized per-channel mean ~0:", out_m.mean(dim=(0, 2, 3)).abs().max().item())
    mine.eval()
    ref.eval()
    print("eval  max diff:", (mine(x) - ref(x)).abs().max().item())

    # 2) 모델 forward shape 확인
    for use_bn in (False, True):
        model = SmallCNN(num_classes=10, use_bn=use_bn)
        out = model(torch.randn(4, 3, 32, 32))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"use_bn={use_bn}: output shape: {out.shape}, params: {n_params}")  # [4, 10]
