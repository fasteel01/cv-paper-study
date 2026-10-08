"""
VGGNet (Simonyan & Zisserman, 2014) — from-scratch PyTorch implementation.

원 논문과의 차이점(현대적 관례로 대체):
  - 구성 C(1x1 conv 포함 VGG-16)는 생략하고 A(11), B(13), D(16), E(19)만 제공
  - 논문의 "얕은 망 사전학습 후 초기화" 대신 Kaiming 초기화 사용
  - 선택 옵션 `batch_norm=True`: conv 뒤 BatchNorm 추가 (논문 시점에는 없던 기법, 기본 False)
  - AdaptiveAvgPool2d로 분류기 입력 크기를 고정 -> 224 이외의 해상도도 허용
  - `pool_size`, `hidden_dim` 인자로 작은 입력(CIFAR 등)용 경량 분류기를 만들 수 있음
    (기본값 7, 4096은 논문과 동일)
"""

import torch
import torch.nn as nn

# 숫자 = 3x3 conv 출력 채널, "M" = 2x2 max-pool(stride 2)
CFGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_features(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_ch = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_ch, v, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """입력: (N, 3, H, W), 출력: (N, num_classes) logits"""

    def __init__(
        self,
        config: str = "D",
        num_classes: int = 1000,
        batch_norm: bool = False,
        dropout: float = 0.5,
        pool_size: int = 7,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.features = make_features(CFGS[config], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Linear(512 * pool_size * pool_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    for cfg in ["A", "B", "D", "E"]:
        model = VGG(cfg, num_classes=1000)
        out = model(dummy)
        print(f"VGG config {cfg}: output shape {tuple(out.shape)}, params {count_params(model) / 1e6:.1f}M")
    # 경량 분류기 (CIFAR-10용)
    small = VGG("A", num_classes=10, batch_norm=True, pool_size=1, hidden_dim=512)
    print("small VGG-11-BN on 32x32:", tuple(small(torch.randn(2, 3, 32, 32)).shape))
