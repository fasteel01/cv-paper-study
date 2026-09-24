"""
AlexNet (Krizhevsky et al., 2012) — from-scratch PyTorch implementation.

원 논문과의 차이점(현대적 관례로 대체):
  - 원 논문의 2-GPU 분할 구조 -> 단일 경로로 통합
  - Local Response Normalization은 옵션으로만 제공(기본 비활성화, 최신 구현들은 보통 생략)
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """입력: (N, 3, 224, 224), 출력: (N, num_classes) logits"""

    def __init__(self, num_classes: int = 1000, use_lrn: bool = False, dropout: float = 0.5):
        super().__init__()

        def lrn():
            return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0) if use_lrn else nn.Identity()

        self.features = nn.Sequential(
            # Conv1: 11x11, stride 4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 5x5
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3-5: 3x3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 224 입력 기준 마지막 feature map: 256 x 6 x 6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(model)
    print("output shape:", out.shape)  # torch.Size([2, 1000])
