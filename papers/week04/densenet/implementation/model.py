"""
DenseNet (Huang et al., 2017) — from-scratch PyTorch implementation.

구성:
  - _DenseLayer : BN-ReLU-Conv1x1(bn_size*k) -> BN-ReLU-Conv3x3(k), 출력을 입력에 concat (DenseNet-B)
  - _DenseBlock : _DenseLayer를 반복. 입력 채널이 층마다 k씩 증가
  - _Transition : BN-ReLU-Conv1x1(theta 배 압축) + AvgPool2x2 (DenseNet-C)

원 논문과의 차이점(현대적 단순화):
  - 메모리 절약용(checkpointing) 구현은 생략 -> 학습 시 메모리를 더 사용
  - dropout은 기본 0 (논문은 증강 없는 데이터셋에서만 0.2 사용)
  - 가중치 초기화는 Kaiming normal (논문의 He 초기화와 동일 계열)
  - 비-BC CIFAR 모델(`bc=False`)도 편의상 bottleneck 1x1 conv는 유지하고 compression만 끈다
    (논문의 순수 DenseNet은 3x3 conv만 사용)
"""

from typing import Sequence

import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter, kernel_size=1, bias=False),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new = self.drop(self.conv(self.bottleneck(x)))
        return torch.cat([x, new], dim=1)  # x_l = [x_0, ..., x_{l-1}, H_l(...)]


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__(
            *[
                _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
                for i in range(num_layers)
            ]
        )
        self.out_channels = in_channels + num_layers * growth_rate


class _Transition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )


class DenseNet(nn.Module):
    """
    Args:
        growth_rate: k
        block_config: 각 dense block의 층 수
        num_init_features: 초기 conv 출력 채널 (보통 2k)
        bn_size: bottleneck 1x1 conv의 출력 = bn_size * k (DenseNet-B, 논문은 4)
        compression: transition의 채널 압축 비율 theta (DenseNet-C, 논문은 0.5; 1.0이면 압축 없음)
        small_inputs: True면 CIFAR형 stem(3x3 conv), False면 ImageNet형 stem(7x7 s2 + maxpool)
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        num_classes: int = 1000,
        small_inputs: bool = False,
    ):
        super().__init__()
        if small_inputs:
            stem = [nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)]
        else:
            stem = [
                nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        self.features = nn.Sequential(*stem)

        ch = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, ch, growth_rate, bn_size, drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            ch = block.out_channels
            if i != len(block_config) - 1:
                out = int(ch * compression)
                self.features.add_module(f"transition{i + 1}", _Transition(ch, out))
                ch = out
        self.features.add_module("norm_final", nn.BatchNorm2d(ch))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)


def densenet121(num_classes: int = 1000, **kw) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes, **kw)


def densenet169(num_classes: int = 1000, **kw) -> DenseNet:
    return DenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes, **kw)


def densenet201(num_classes: int = 1000, **kw) -> DenseNet:
    return DenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes, **kw)


def densenet161(num_classes: int = 1000, **kw) -> DenseNet:
    return DenseNet(48, (6, 12, 36, 24), 96, num_classes=num_classes, **kw)


def densenet_cifar(depth: int = 100, growth_rate: int = 12, bc: bool = True, num_classes: int = 10, **kw) -> DenseNet:
    """CIFAR용 3-block DenseNet. BC: (depth-4)/6 층/블록(층당 conv 2개), 비-BC: (depth-4)/3."""
    if bc:
        n = (depth - 4) // 6
        return DenseNet(growth_rate, (n,) * 3, 2 * growth_rate, bn_size=4, compression=0.5,
                        num_classes=num_classes, small_inputs=True, **kw)
    n = (depth - 4) // 3
    return DenseNet(growth_rate, (n,) * 3, 16, bn_size=4, compression=1.0,
                    num_classes=num_classes, small_inputs=True, **kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = densenet121()
    out = model(torch.randn(2, 3, 224, 224))
    print("DenseNet-121 output shape:", out.shape, f"params: {count_params(model) / 1e6:.2f}M")  # (2, 1000)

    small = densenet_cifar(100, 12, bc=True)
    out = small(torch.randn(2, 3, 32, 32))
    print("DenseNet-BC-100 (CIFAR) output shape:", out.shape, f"params: {count_params(small) / 1e6:.2f}M")  # (2, 10)
