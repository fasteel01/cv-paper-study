"""
ResNet (He et al., 2015) — from-scratch PyTorch implementation.

구성: BasicBlock(18/34), Bottleneck(50/101/152), ResNet, resnet18/34/50/101/152 팩토리.

원 논문과의 차이점(현대적 단순화):
  - shortcut 차원 불일치 시 항상 1x1 conv + BN 투영(option B)만 사용 (option A 0-패딩 항등은 생략)
  - Bottleneck의 stride를 3x3 conv에 둠 (torchvision 관례, ResNet v1.5; 원 논문은 첫 1x1에 둠)
  - residual=False 로 shortcut을 끄면 plain 망이 되어 degradation 비교 가능 (실험용 옵션)
  - cifar_stem=True: 32x32 입력용 3x3 stem, maxpool 제거 (논문의 CIFAR 실험 구조와 유사한 관례)
  - zero_init_residual: 마지막 BN의 gamma를 0으로 초기화하는 옵션 (원 논문에는 없음)
  - 가중치 초기화는 He(Kaiming) 초기화를 사용
"""

import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, ch, stride=1, residual=True):
        super().__init__()
        self.conv1 = conv3x3(in_ch, ch, stride)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = conv3x3(ch, ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        self.shortcut = self._make_shortcut(in_ch, ch * self.expansion, stride) if residual else None

    @staticmethod
    def _make_shortcut(in_ch, out_ch, stride):
        if stride != 1 or in_ch != out_ch:
            return nn.Sequential(conv1x1(in_ch, out_ch, stride), nn.BatchNorm2d(out_ch))
        return nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)  # F(x) + x
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, ch, stride=1, residual=True):
        super().__init__()
        out_ch = ch * self.expansion
        self.conv1 = conv1x1(in_ch, ch)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = conv3x3(ch, ch, stride)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = conv1x1(ch, out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        self.shortcut = BasicBlock._make_shortcut(in_ch, out_ch, stride) if residual else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    """입력: (N, 3, H, W), 출력: (N, num_classes) logits"""

    def __init__(self, block, layers, num_classes=1000, residual=True, cifar_stem=False,
                 zero_init_residual=False):
        super().__init__()
        self.in_ch = 64
        if cifar_stem:
            self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        stages = []
        for i, n_blocks in enumerate(layers):
            stages.append(self._make_stage(block, 64 * 2 ** i, n_blocks, 1 if i == 0 else 2, residual))
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_stage(self, block, ch, n_blocks, stride, residual):
        blocks = []
        for i in range(n_blocks):
            blocks.append(block(self.in_ch, ch, stride if i == 0 else 1, residual))
            self.in_ch = ch * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet18(**kw):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kw)


def resnet34(**kw):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kw)


def resnet50(**kw):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kw)


def resnet101(**kw):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kw)


def resnet152(**kw):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kw)


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    for name, fn in [("resnet18", resnet18), ("resnet34", resnet34), ("resnet50", resnet50),
                     ("resnet101", resnet101), ("resnet152", resnet152)]:
        model = fn(num_classes=1000).eval()
        with torch.no_grad():
            out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape {tuple(out.shape)}, params {n_params / 1e6:.2f}M")  # (2, 1000)

    # CIFAR용 + plain 비교 옵션
    small = resnet18(num_classes=10, cifar_stem=True, residual=False)
    print("plain18 cifar output shape:", small(torch.randn(2, 3, 32, 32)).shape)  # [2, 10]
