"""
GoogLeNet / Inception v1 (Szegedy et al., 2014) — from-scratch PyTorch implementation.

원 논문과의 차이점(현대적 관례로 대체):
  - Local Response Normalization 층은 생략
  - 보조 분류기의 5x5 avgpool(stride 3) 대신 AdaptiveAvgPool2d(4) 사용
    (224 입력에서는 논문과 동일하게 4x4 특징 맵이 됨, 다른 해상도도 허용)
  - 최종 7x7 avgpool 대신 AdaptiveAvgPool2d(1) 사용
  - Kaiming 초기화 사용
  - 보조 분류기는 학습 모드 + aux_logits=True일 때만 출력 (추론 시 제거, 논문과 동일한 의도)
"""

import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Inception(nn.Module):
    """4개 가지를 병렬 적용 후 채널 concat. 출력 채널 = n1 + n3 + n5 + pool_proj"""

    def __init__(self, in_ch, n1, n3r, n3, n5r, n5, pool_proj):
        super().__init__()
        self.branch1 = ConvReLU(in_ch, n1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvReLU(in_ch, n3r, kernel_size=1),
            ConvReLU(n3r, n3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvReLU(in_ch, n5r, kernel_size=1),
            ConvReLU(n5r, n5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvReLU(in_ch, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_ch, num_classes, dropout=0.7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.conv = ConvReLU(in_ch, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv(self.pool(x))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class GoogLeNet(nn.Module):
    """입력: (N, 3, 224, 224).
    출력: 학습 모드 & aux_logits=True -> (main, aux1, aux2), 그 외 -> main logits
    """

    def __init__(self, num_classes: int = 1000, aux_logits: bool = True, dropout: float = 0.4):
        super().__init__()
        self.aux_logits = aux_logits

        self.stem = nn.Sequential(
            ConvReLU(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            ConvReLU(64, 64, kernel_size=1),
            ConvReLU(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )

        # (in, n1, n3r, n3, n5r, n5, pool_proj)
        self.inc3a = Inception(192, 64, 96, 128, 16, 32, 32)  # -> 256
        self.inc3b = Inception(256, 128, 128, 192, 32, 96, 64)  # -> 480
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inc4a = Inception(480, 192, 96, 208, 16, 48, 64)  # -> 512
        self.inc4b = Inception(512, 160, 112, 224, 24, 64, 64)  # -> 512
        self.inc4c = Inception(512, 128, 128, 256, 24, 64, 64)  # -> 512
        self.inc4d = Inception(512, 112, 144, 288, 32, 64, 64)  # -> 528
        self.inc4e = Inception(528, 256, 160, 320, 32, 128, 128)  # -> 832
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inc5a = Inception(832, 256, 160, 320, 32, 128, 128)  # -> 832
        self.inc5b = Inception(832, 384, 192, 384, 48, 128, 128)  # -> 1024

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)  # 4a 출력에 연결
            self.aux2 = AuxClassifier(528, num_classes)  # 4d 출력에 연결
        else:
            self.aux1 = self.aux2 = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool3(self.inc3b(self.inc3a(x)))

        x = self.inc4a(x)
        aux1 = self.aux1(x) if (self.aux1 is not None and self.training) else None
        x = self.inc4d(self.inc4c(self.inc4b(x)))
        aux2 = self.aux2(x) if (self.aux2 is not None and self.training) else None
        x = self.pool4(self.inc4e(x))

        x = self.inc5b(self.inc5a(x))
        x = torch.flatten(self.avgpool(x), 1)
        out = self.fc(self.dropout(x))

        if self.training and self.aux_logits:
            return out, aux1, aux2
        return out


if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000)
    dummy = torch.randn(2, 3, 224, 224)

    model.train()
    main, a1, a2 = model(dummy)
    print("train mode -> main:", tuple(main.shape), "aux1:", tuple(a1.shape), "aux2:", tuple(a2.shape))

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print("eval mode  -> output shape:", tuple(out.shape))  # (2, 1000)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Inception 3a 채널 확인: 64 + 128 + 32 + 32 = 256
    print("inception 3a out:", tuple(model.inc3a(model.stem(dummy)).shape))
