"""
DeepLabv3 (Chen et al., 2017) — from-scratch PyTorch implementation.

구성: ResNet(bottleneck, atrous) backbone + ASPP(1x1, 3x3 atrous x3, image pooling) + classifier.

원 논문과의 차이점 / 단순화:
  - backbone 은 기본 ResNet-50 구성 (3,4,6,3). 논문은 ResNet-101 (블록 수 (3,4,23,3)) 사용. `layers` 인자로 변경 가능.
  - 사전학습(ImageNet) 가중치는 사용하지 않고 무작위 초기화.
  - 논문의 ResNet-101 기반 "deep stem"/세부 초기화는 생략하고 표준 ResNet stem(7x7 conv + maxpool) 사용.
  - 다중 스케일 추론, COCO 사전학습, DenseCRF 등은 포함하지 않음 (DeepLabv3 는 CRF 미사용).
  - ASPP rate 는 output_stride 에 따라 (6,12,18) [OS=16] 또는 (12,24,36) [OS=8] 로 자동 설정.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """1x1 -> 3x3(dilated) -> 1x1 residual unit. expansion = 4."""

    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, dilation=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class AtrousResNet(nn.Module):
    """
    output_stride 에 따라 layer3/layer4 의 stride 를 dilation 으로 대체한 ResNet.
      OS=32: 일반 ResNet
      OS=16: layer4 stride 1, dilation 2
      OS=8 : layer3 stride 1 dilation 2, layer4 stride 1 dilation 4
    multi_grid: layer4 의 각 블록 dilation 에 곱할 배수 (예: (1,2,4)).
    """

    def __init__(self, layers: Sequence[int] = (3, 4, 6, 3), output_stride: int = 16,
                 multi_grid: Sequence[int] = (1, 1, 1)):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("output_stride must be 8, 16 or 32")
        strides = {32: (1, 2, 2, 2), 16: (1, 2, 2, 1), 8: (1, 2, 1, 1)}[output_stride]
        dilations = {32: (1, 1, 1, 1), 16: (1, 1, 1, 2), 8: (1, 1, 2, 4)}[output_stride]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.in_ch = 64
        self.layer1 = self._make_layer(64, layers[0], strides[0], dilations[0])
        self.layer2 = self._make_layer(128, layers[1], strides[1], dilations[1])
        self.layer3 = self._make_layer(256, layers[2], strides[2], dilations[2])
        # layer4: multi-grid 적용 (블록 수가 3 이 아니면 grid 를 순환 사용)
        self.layer4 = self._make_layer(512, layers[3], strides[3], dilations[3], multi_grid=multi_grid)
        self.out_channels = 512 * Bottleneck.expansion

    def _make_layer(self, mid_ch, blocks, stride, dilation, multi_grid=None):
        grid = list(multi_grid) if multi_grid is not None else [1] * blocks
        units: List[nn.Module] = []
        for i in range(blocks):
            d = dilation * grid[i % len(grid)]
            units.append(Bottleneck(self.in_ch, mid_ch, stride=stride if i == 0 else 1, dilation=d))
            self.in_ch = mid_ch * Bottleneck.expansion
        return nn.Sequential(*units)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def conv_bn_relu(in_ch, out_ch, k=1, dilation=1):
    pad = 0 if k == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (v3): 1x1 + 3 atrous 3x3 + image-level pooling -> concat -> 1x1."""

    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, 1)] + [conv_bn_relu(in_ch, out_ch, 3, dilation=r) for r in rates]
        )
        # image pooling 분기: GAP -> 1x1 conv -> (bilinear upsample 은 forward 에서)
        # BN 은 1x1 공간 크기에서 batch=1 일 때 학습이 불가하므로 주의 (batch>=2 필요)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n = len(self.branches) + 1
        self.project = nn.Sequential(conv_bn_relu(out_ch * n, out_ch, 1), nn.Dropout(dropout))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        pooled = F.interpolate(self.image_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        outs.append(pooled)
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3(nn.Module):
    """입력: (N, 3, H, W) -> 출력: (N, num_classes, H, W) logits (입력 해상도로 bilinear 업샘플)"""

    def __init__(self, num_classes=21, output_stride=16, layers=(3, 4, 6, 3), multi_grid=(1, 1, 1),
                 aspp_channels=256):
        super().__init__()
        self.backbone = AtrousResNet(layers, output_stride, multi_grid)
        rates = {16: (6, 12, 18), 8: (12, 24, 36), 32: (3, 6, 9)}[output_stride]
        self.aspp = ASPP(self.backbone.out_channels, aspp_channels, rates)
        self.classifier = nn.Conv2d(aspp_channels, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feat = self.backbone(x)
        logits = self.classifier(self.aspp(feat))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 257, 257)
    for os_ in (32, 16, 8):
        m = DeepLabV3(num_classes=21, output_stride=os_, multi_grid=(1, 2, 4) if os_ != 32 else (1, 1, 1))
        m.eval()
        with torch.no_grad():
            feat = m.backbone(x)
            y = m(x)
        print(f"OS={os_:>2}: backbone feat {tuple(feat.shape)} -> logits {tuple(y.shape)}")
        assert y.shape == (2, 21, 257, 257)
    n_params = sum(p.numel() for p in DeepLabV3().parameters())
    print(f"params (OS=16, ResNet-50 style): {n_params/1e6:.1f}M")
