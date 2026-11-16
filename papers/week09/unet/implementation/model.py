"""
U-Net (Ronneberger, Fischer, Brox, 2015) — from-scratch PyTorch implementation.

구성 (논문 Figure 1)
  - Contracting path(인코더): [3x3 conv + ReLU] x 2 -> 2x2 max pool(stride 2) 를 4번 반복, 채널은 매 단계 2배 (64,128,256,512)
  - Bottleneck: [3x3 conv + ReLU] x 2, 1024 채널 (논문: 이 끝에 dropout)
  - Expansive path(디코더): 2x2 up-convolution(전치 합성곱, 채널 절반) -> 같은 해상도 인코더 특징맵을
    (필요 시 중앙 crop 해서) 채널 방향으로 concatenate -> [3x3 conv + ReLU] x 2. 4번 반복
  - 마지막 1x1 conv 로 클래스 수만큼의 채널(논문: 2 = 세포/배경) 출력
  - 가중치 초기화: 표준편차 sqrt(2/N) 의 가우시안 (N = k*k*in_channels) = Kaiming(fan_in) 정규분포

두 가지 padding 모드
  - padding=False (기본, 논문 그대로): 3x3 conv 에 패딩이 없어(valid) 매 conv 마다 가장자리 1픽셀씩 줄어듦.
    입력 572x572 -> 출력 388x388. skip 연결은 인코더 특징맵을 중앙 crop 해서 맞춤.
    (논문은 큰 이미지를 overlap-tile 전략으로 처리하고 경계는 거울 반사(mirror) 패딩으로 채움)
  - padding=True: 'same' 패딩. 입출력 크기가 같고 crop 이 필요 없음 (현대 구현에서 흔한 단순화)

원 논문과의 차이(단순화):
  - 기본 입력 채널 1(grayscale 현미경 영상), 출력 채널 2 이지만 인자로 변경 가능
  - batch_norm 옵션(기본 꺼짐): 논문은 BN 이전 시대(2015)라 사용하지 않음. 켜면 conv(bias 없음)-BN-ReLU
  - dropout 은 bottleneck 뒤 Dropout2d 로 구현(기본 0 = 사용 안 함, 논문은 사용)
  - 업샘플은 논문과 같은 2x2 ConvTranspose2d (현대 구현은 bilinear upsample + conv 로 대체하기도 함)
  - 탄성 변형(elastic deformation) 증강, 가중치 맵 손실, overlap-tile 추론은 model.py 가 아닌 train.py/요약 참고

입력: (N, in_channels, H, W) -> 출력: (N, num_classes, H', W') logits (padding=True 이면 H'=H, W'=W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop(x: torch.Tensor, size) -> torch.Tensor:
    """(.., H, W) 텐서를 (th, tw) 로 중앙 crop. skip connection 의 크기를 디코더에 맞출 때 사용."""
    h, w = x.shape[-2:]
    th, tw = size
    top, left = (h - th) // 2, (w - tw) // 2
    return x[..., top : top + th, left : left + tw]


class DoubleConv(nn.Module):
    """[3x3 conv -> (BN) -> ReLU] x 2"""

    def __init__(self, in_c: int, out_c: int, padding: bool, batch_norm: bool):
        super().__init__()
        p = 1 if padding else 0
        layers = []
        for i in range(2):
            layers.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=p, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 64, depth: int = 4,
                 padding: bool = False, batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        self.depth, self.padding = depth, padding
        chs = [base * 2**i for i in range(depth + 1)]  # 64,128,256,512,1024

        self.down = nn.ModuleList()
        c = in_channels
        for out_c in chs[:-1]:
            self.down.append(DoubleConv(c, out_c, padding, batch_norm))
            c = out_c
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1], padding, batch_norm)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for out_c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(out_c * 2, out_c, kernel_size=2, stride=2))  # 채널 절반
            self.up_conv.append(DoubleConv(out_c * 2, out_c, padding, batch_norm))  # concat 후 2*out_c -> out_c
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # std = sqrt(2 / fan_in)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)  # 풀링 전의 특징맵을 skip 으로 저장
            x = self.pool(x)
        x = self.drop(self.bottleneck(x))
        for up, conv, skip in zip(self.up, self.up_conv, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                skip = center_crop(skip, x.shape[-2:])  # valid 모드: 인코더 특징맵이 더 크므로 중앙을 잘라냄
            x = conv(torch.cat([skip, x], dim=1))  # 채널 concat (FCN 의 합산과 다름)
        return self.head(x)

    def output_size(self, size: int) -> int:
        """한 변의 입력 크기 -> 출력 크기. valid 모드에서 풀링 직전 크기가 홀수이면 사용할 수 없는 입력이라 에러."""
        if self.padding:
            if size % (2**self.depth):
                raise ValueError(f"padding=True 에서는 입력이 {2**self.depth} 의 배수여야 skip 크기가 맞습니다: {size}")
            return size
        s = size
        for _ in range(self.depth):
            s -= 4  # conv 2번
            if s % 2:
                raise ValueError(f"입력 {size}: 풀링 직전 크기 {s} 가 홀수입니다")
            s //= 2
        s -= 4  # bottleneck
        for _ in range(self.depth):
            s = s * 2 - 4
        if s <= 0:
            raise ValueError("입력이 너무 작습니다")
        return s


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 논문 설정: 572x572 입력 -> 388x388 출력 (valid conv)
    model = UNet(in_channels=1, num_classes=2, padding=False).eval()
    x = torch.randn(1, 1, 572, 572)
    with torch.no_grad():
        out = model(x)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"paper U-Net (valid conv): input {tuple(x.shape)} -> output {tuple(out.shape)} | params {n_params:.1f}M")
    assert out.shape == (1, 2, 388, 388) and model.output_size(572) == 388

    # 2) same-padding 변형: 입출력 크기 동일
    model2 = UNet(in_channels=3, num_classes=5, padding=True, batch_norm=True).eval()
    x2 = torch.randn(2, 3, 128, 192)
    with torch.no_grad():
        out2 = model2(x2)
    print(f"same-padding + BN:       input {tuple(x2.shape)} -> output {tuple(out2.shape)}")
    assert out2.shape == (2, 5, 128, 192)

    # 3) 중간 shape 추적: 인코더 skip 과 디코더 up-conv 출력이 어떻게 맞춰지는지
    shapes, h = [], 572
    for _ in range(4):
        h -= 4
        shapes.append(h)
        h //= 2
    print("encoder skip sizes (572 입력):", shapes, "| bottleneck:", h - 4)
