"""
FCN (Long, Shelhamer, Darrell, 2015) — from-scratch PyTorch implementation.

구성
  - VGG-16 백본 (conv 13개 + pool 5개). 분류기의 FC 층을 합성곱으로 변환:
      fc6 = 7x7 conv (4096), fc7 = 1x1 conv (4096), + dropout   -> 임의 크기 입력 가능한 "fully convolutional" 구조
  - score 층: 1x1 conv 로 클래스 수(기본 21 = VOC 20 + 배경) 채널의 히트맵 생성
  - FCN-32s : pool5 이후 score(stride 32) 를 32배 업샘플 한 번에 복원
  - FCN-16s : score(32) 를 2배 업샘플 + pool4 위의 1x1 score(16) 를 합산 -> 16배 업샘플
  - FCN-8s  : 위 결과를 2배 업샘플 + pool3 위의 1x1 score(8) 를 합산 -> 8배 업샘플
  - 업샘플은 학습 가능한 전치 합성곱(ConvTranspose2d), bilinear 보간 가중치로 초기화

원 논문과의 차이(단순화):
  - 논문은 conv1_1 에 padding=100 을 주고 이후 crop 층으로 크기를 맞춤. 여기서는 conv1_1 을 padding=1,
    fc6 를 padding=3 으로 두어 "same" 규칙을 만들고, 크기가 어긋나는 경우(입력이 32 의 배수가 아닐 때)만
    F.interpolate / 중앙 crop 으로 맞춤. 입력이 32 의 배수이면 crop 이 필요 없음
  - 논문은 32s -> 16s -> 8s 순으로 이전 단계 가중치에서 이어 학습. 여기서는 stage 인자로 구조만 선택하고
    각각 처음부터 학습 (skip score 층과 최종 score 층은 논문처럼 0 으로 초기화).
    사전학습(ImageNet VGG-16) 가중치 로딩 없음(무작위 초기화)
  - BatchNorm 없음 (원 VGG-16 과 동일)

입력: (N, 3, H, W) -> 출력: (N, num_classes, H, W) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    """ConvTranspose2d 가중치를 bilinear 보간 필터로 초기화 (채널별 독립: 대각 구조)."""
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og = torch.arange(kernel_size, dtype=torch.float32)
    filt1d = 1 - torch.abs(og - center) / factor
    filt = filt1d[:, None] * filt1d[None, :]
    weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
    for c in range(min(in_channels, out_channels)):
        weight[c, c] = filt
    return weight


def make_upsample(num_classes: int, factor: int) -> nn.ConvTranspose2d:
    """factor 배 업샘플하는 전치 합성곱: kernel = 2*factor, stride = factor, padding = factor/2."""
    up = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2 * factor, stride=factor,
                            padding=factor // 2, bias=False)
    with torch.no_grad():
        up.weight.copy_(bilinear_kernel(num_classes, num_classes, 2 * factor))
    return up


def center_crop(x: torch.Tensor, size) -> torch.Tensor:
    h, w = x.shape[-2:]
    th, tw = size
    top, left = (h - th) // 2, (w - tw) // 2
    return x[..., top : top + th, left : left + tw]


def _vgg_block(in_c, out_c, n):
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class FCN(nn.Module):
    """stage: '32s' | '16s' | '8s'"""

    def __init__(self, num_classes: int = 21, stage: str = "8s", fc_channels: int = 4096, dropout: float = 0.5):
        super().__init__()
        assert stage in ("32s", "16s", "8s")
        self.stage = stage
        self.num_classes = num_classes

        # VGG-16 feature extractor, 각 pool 이후가 stride 2,4,8,16,32
        self.block1 = _vgg_block(3, 64, 2)
        self.block2 = _vgg_block(64, 128, 2)
        self.block3 = _vgg_block(128, 256, 3)   # -> pool3, stride 8
        self.block4 = _vgg_block(256, 512, 3)   # -> pool4, stride 16
        self.block5 = _vgg_block(512, 512, 3)   # -> pool5, stride 32

        # FC -> conv (convolutionalization)
        self.fc6 = nn.Sequential(nn.Conv2d(512, fc_channels, 7, padding=3), nn.ReLU(inplace=True), nn.Dropout2d(dropout))
        self.fc7 = nn.Sequential(nn.Conv2d(fc_channels, fc_channels, 1), nn.ReLU(inplace=True), nn.Dropout2d(dropout))
        self.score_fr = nn.Conv2d(fc_channels, num_classes, 1)  # stride 32 score

        if stage == "32s":
            self.up32 = make_upsample(num_classes, 32)
        else:
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
            self.up2_a = make_upsample(num_classes, 2)  # 32 -> 16
            if stage == "16s":
                self.up16 = make_upsample(num_classes, 16)
            else:
                self.score_pool3 = nn.Conv2d(256, num_classes, 1)
                self.up2_b = make_upsample(num_classes, 2)  # 16 -> 8
                self.up8 = make_upsample(num_classes, 8)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # skip 연결용 score 층은 0 으로 초기화: 초기에는 이전 단계(예: 32s) 예측을 그대로 유지하고 skip 이 점진적으로 기여
        for name in ("score_pool4", "score_pool3"):
            if hasattr(self, name):
                nn.init.zeros_(getattr(self, name).weight)
                nn.init.zeros_(getattr(self, name).bias)
        nn.init.zeros_(self.score_fr.weight)
        nn.init.zeros_(self.score_fr.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]
        p3 = self.block3(self.block2(self.block1(x)))
        p4 = self.block4(p3)
        p5 = self.block5(p4)
        score = self.score_fr(self.fc7(self.fc6(p5)))  # stride 32

        if self.stage == "32s":
            out = self.up32(score)
        else:
            up = self.up2_a(score)
            s4 = self.score_pool4(p4)
            up = self._match(up, s4.shape[-2:])
            fused = up + s4  # stride 16 에서 합산 (element-wise sum)
            if self.stage == "16s":
                out = self.up16(fused)
            else:
                up = self.up2_b(fused)
                s3 = self.score_pool3(p3)
                up = self._match(up, s3.shape[-2:])
                out = self.up8(up + s3)  # stride 8 에서 합산

        return self._match(out, in_hw)

    @staticmethod
    def _match(x: torch.Tensor, size) -> torch.Tensor:
        """공간 크기 보정: 크면 중앙 crop, 작으면 bilinear 로 맞춤 (32 의 배수 입력에서는 no-op)."""
        if x.shape[-2:] == tuple(size):
            return x
        if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
            return center_crop(x, size)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    for stage in ("32s", "16s", "8s"):
        model = FCN(num_classes=21, stage=stage).eval()
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"FCN-{stage}: output {tuple(out.shape)} | params {n_params:.1f}M")
        assert out.shape == (2, 21, 224, 224)

    # fully convolutional: 다른 크기(32 의 배수가 아닌 것 포함)의 입력도 그대로 처리
    model = FCN(num_classes=21, stage="8s").eval()
    with torch.no_grad():
        for hw in [(320, 480), (250, 333)]:
            o = model(torch.randn(1, 3, *hw))
            print(f"input {hw} -> output {tuple(o.shape)}")
            assert o.shape[-2:] == hw

    # 전치 합성곱이 bilinear 보간과 같은 결과를 내는지 확인 (가장자리 제외)
    up = make_upsample(1, 4)
    t = torch.randn(1, 1, 8, 8)
    a = up(t)
    b = F.interpolate(t, scale_factor=4, mode="bilinear", align_corners=False)
    print("deconv(bilinear init) vs F.interpolate, 안쪽 영역 최대 오차: %.2e" % (a[..., 4:-4, 4:-4] - b[..., 4:-4, 4:-4]).abs().max().item())
