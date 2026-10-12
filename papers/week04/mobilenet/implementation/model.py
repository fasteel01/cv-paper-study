"""
MobileNet v1 (Howard et al., 2017) — from-scratch PyTorch implementation.

구성:
  - DepthwiseSeparableConv : 3x3 depthwise(groups=C) + BN + ReLU -> 1x1 pointwise + BN + ReLU
  - StandardConvBlock      : 비교용 3x3 표준 conv + BN + ReLU (separable=False)
  - MobileNetV1            : 첫 3x3 s2 conv + 13개 블록 + GAP + FC. width_mult(alpha) 지원

원 논문과의 차이점(현대적 단순화):
  - 해상도 배율 rho는 별도 인자 없이 입력 이미지 크기를 바꿔 적용 (global avg pool은 AdaptiveAvgPool)
  - 채널 수는 width_mult 적용 후 8의 배수로 반올림 (최소 8)
  - 마지막 FC 앞 dropout은 옵션 (논문 고유 설정 아님)
  - 학습 하이퍼파라미터(RMSprop 등)는 train.py에서 단순한 설정으로 대체
"""

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class StandardConvBlock(nn.Sequential):
    """같은 입출력 형태를 갖는 표준 3x3 conv (비교 실험용)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# (출력 채널, stride) — 논문 Table 1의 dw/pw 쌍
CFG = [
    (64, 1),
    (128, 2), (128, 1),
    (256, 2), (256, 1),
    (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
    (1024, 2), (1024, 1),
]


class MobileNetV1(nn.Module):
    """입력: (N, 3, H, W), 출력: (N, num_classes) logits. 224 입력이면 마지막 특징 맵은 7x7."""

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        separable: bool = True,
        first_stride: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        block = DepthwiseSeparableConv if separable else StandardConvBlock
        ch = _make_divisible(32 * width_mult)
        layers = [
            nn.Conv2d(3, ch, kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        ]
        for out, stride in CFG:
            out = _make_divisible(out * width_mult)
            layers.append(block(ch, out, stride))
            ch = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(self.dropout(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for name, kw in [
        ("MobileNet 1.0 (separable)", dict()),
        ("MobileNet 0.5 (separable)", dict(width_mult=0.5)),
        ("Standard-conv variant", dict(separable=False)),
    ]:
        model = MobileNetV1(num_classes=1000, **kw)
        out = model(x)
        print(f"{name}: output {tuple(out.shape)}, params {count_params(model) / 1e6:.2f}M")  # (2, 1000)
