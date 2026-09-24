"""
LeNet-5 (LeCun et al., 1998) — from-scratch PyTorch implementation.

원 논문과의 차이점(현대적 관례로 대체):
  - 활성화 함수: tanh -> ReLU
  - 출력층: Euclidean RBF -> Linear + Softmax(CrossEntropyLoss에서 처리)
  - Subsampling: 논문의 학습 가능한 스칼라+bias average pooling 대신 표준 AvgPool2d 사용
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """입력: (N, 1, 32, 32) grayscale 이미지, 출력: (N, num_classes) logits"""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            # C1: 5x5 conv, 1 -> 6 channels, 28x28 출력
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # S2: 2x2 average pooling, 14x14 출력
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 5x5 conv, 6 -> 16 channels, 10x10 출력
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # S4: 2x2 average pooling, 5x5 출력
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            # C5: 5x5 conv == fully connected (16*5*5 -> 120)
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            # F6
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = LeNet5(num_classes=10)
    dummy = torch.randn(2, 1, 32, 32)
    out = model(dummy)
    print(model)
    print("output shape:", out.shape)  # torch.Size([2, 10])
