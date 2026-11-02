"""
Feature Pyramid Network (Lin et al., 2017) — from-scratch PyTorch implementation.

구성
  - ResNet-50 백본 (C2~C5, stride 4/8/16/32)
  - FPN: lateral 1x1 conv + top-down(2x nearest upsample, 원소별 합) + 3x3 smoothing conv -> P2~P5, P6 = P5 의 stride-2 subsample
  - RPN head: 피라미드 모든 층에서 가중치를 공유, 레벨마다 앵커 "크기" 하나(32~512) x 종횡비 3개
  - RoI -> 피라미드 레벨 할당 k = floor(k0 + log2(sqrt(wh) / 224)), k0 = 4, [2,5] 로 clamp
  - 멀티레벨 RoI 특징 추출 + Fast R-CNN box head (2 FC 1024)

원 논문과의 차이(단순화):
  - RoI 특징 추출에 논문의 RoIPool 대신 torchvision.ops.roi_align 사용 (Mask R-CNN 이후 표준)
  - proposal 단계의 NMS 는 torchvision.ops.nms 사용. 나머지 박스 연산(IoU/encode/decode/앵커)은 직접 구현
  - 백본은 BatchNorm 을 일반 BN 으로 사용 (논문은 학습 중 고정(frozen) BN 사용), stride 는 3x3 conv 에 둠(ResNet v1.5)
  - 사전학습 가중치 로딩 없음 (무작위 초기화). 이미지 크기는 32 의 배수를 가정
  - P6 은 RPN 전용(RoI head 에서는 P2~P5 만 사용), 논문과 동일

좌표 규약: 픽셀 단위 (xmin, ymin, xmax, ymax).
"""

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align


# ---------------------------------------------------------------- box utils
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode_boxes(gt: torch.Tensor, ref: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """ref(앵커/proposal) 기준 gt 의 (tx, ty, tw, th)."""
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    wx, wy, ww, wh = weights
    return torch.stack([wx * (gx - rx) / rw, wy * (gy - ry) / rh, ww * torch.log(gw / rw), wh * torch.log(gh / rh)], dim=1)


def decode_boxes(deltas: torch.Tensor, ref: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    wx, wy, ww, wh = weights
    clip = math.log(1000.0 / 16)
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw, dh = (deltas[:, 2] / ww).clamp(max=clip), (deltas[:, 3] / wh).clamp(max=clip)
    cx, cy = rx + dx * rw, ry + dy * rh
    w, h = rw * torch.exp(dw), rh * torch.exp(dh)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=1)


# ---------------------------------------------------------------- ResNet-50 backbone
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, mid_c, stride=1):
        super().__init__()
        out_c = mid_c * self.expansion
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


class ResNet50Backbone(nn.Module):
    """이미지 -> [C2, C3, C4, C5] (채널 256/512/1024/2048, stride 4/8/16/32)"""

    out_channels = (256, 512, 1024, 2048)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        cfg, in_c, stages = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)], 64, []
        for mid, n, stride in cfg:
            blocks = []
            for b in range(n):
                blocks.append(Bottleneck(in_c, mid, stride if b == 0 else 1))
                in_c = mid * Bottleneck.expansion
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


# ---------------------------------------------------------------- FPN
class FPN(nn.Module):
    """[C2..C5] -> [P2..P5, P6], 모두 out_channels(256) 채널."""

    def __init__(self, in_channels: Sequence[int] = (256, 512, 1024, 2048), out_channels: int = 256, extra_p6: bool = True):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])  # 채널 통일
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])  # aliasing 완화
        self.extra_p6 = extra_p6
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down: 가장 깊은 층에서 시작해 2x 업샘플 + lateral 을 원소별로 더함
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        outs = [s(l) for s, l in zip(self.smooth, lat)]
        if self.extra_p6:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))  # P6: P5 의 stride-2 subsampling
        return outs


# ---------------------------------------------------------------- anchors + RPN
def generate_anchors(feature_shapes, strides, sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1.0, 2.0), device="cpu"):
    """레벨별 앵커. 반환: (총 앵커 수, 4) 와 레벨별 앵커 수 리스트.
    순서: level -> row -> col -> ratio. RPN head 출력 flatten 순서와 일치. ratio = h / w."""
    all_anchors, counts = [], []
    for (h, w), stride, size in zip(feature_shapes, strides, sizes):
        r = torch.tensor(ratios, dtype=torch.float32, device=device)
        ws, hs = size / torch.sqrt(r), size * torch.sqrt(r)
        base = torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)  # (A,4)
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 1, 4)  # (HW,1,4)
        a = (centers + base[None]).reshape(-1, 4)
        all_anchors.append(a)
        counts.append(a.size(0))
    return torch.cat(all_anchors), counts


class RPNHead(nn.Module):
    """3x3 conv + (objectness 1x1, box 1x1). 모든 피라미드 레벨에서 가중치 공유."""

    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        objs, regs = [], []
        for f in feats:
            t = F.relu(self.conv(f))
            n = t.size(0)
            objs.append(self.cls(t).permute(0, 2, 3, 1).reshape(n, -1))  # (N, H*W*A)
            regs.append(self.reg(t).permute(0, 2, 3, 1).reshape(n, -1, 4))  # (N, H*W*A, 4)
        return torch.cat(objs, dim=1), torch.cat(regs, dim=1)


# ---------------------------------------------------------------- RoI -> level, multi-level pooling, box head
def map_rois_to_levels(boxes: torch.Tensor, k0: int = 4, canonical: float = 224.0, k_min: int = 2, k_max: int = 5):
    """논문 식 (1): k = floor(k0 + log2(sqrt(wh) / 224)). 반환: 0..(k_max-k_min) 의 레벨 인덱스 (0 = P2)."""
    wh = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    k = torch.floor(k0 + torch.log2(torch.sqrt(wh) / canonical + 1e-8))
    return (k.clamp(k_min, k_max) - k_min).long()


class MultiLevelRoIAlign(nn.Module):
    """RoI 를 크기에 맞는 레벨(P2~P5)에서 7x7 로 추출."""

    def __init__(self, output_size=7, strides=(4, 8, 16, 32), sampling_ratio=2):
        super().__init__()
        self.output_size, self.strides, self.sampling_ratio = output_size, strides, sampling_ratio

    def forward(self, feats: List[torch.Tensor], rois: List[torch.Tensor]):
        """feats: [P2..P5], rois: 이미지별 (R_i,4). 반환 (sum R_i, C, 7, 7) — 이미지 순서대로."""
        flat = torch.cat([torch.cat([torch.full((len(r), 1), float(i), device=r.device), r], dim=1) for i, r in enumerate(rois)])
        levels = map_rois_to_levels(flat[:, 1:])
        out = feats[0].new_zeros(flat.size(0), feats[0].size(1), self.output_size, self.output_size)
        for lvl, (f, s) in enumerate(zip(feats, self.strides)):
            idx = (levels == lvl).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out[idx] = roi_align(f, flat[idx], self.output_size, spatial_scale=1.0 / s,
                                 sampling_ratio=self.sampling_ratio, aligned=True)
        return out


class BoxHead(nn.Module):
    """Fast R-CNN head: flatten -> FC1024 -> FC1024 -> (cls, class-specific box)."""

    def __init__(self, in_channels=256, pool=7, num_classes=21, hidden=1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * pool * pool, hidden)
        self.fc7 = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, num_classes)  # 0 = 배경
        self.reg = nn.Linear(hidden, num_classes * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.reg.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        x = F.relu(self.fc6(x.flatten(1)))
        x = F.relu(self.fc7(x))
        return self.cls(x), self.reg(x)


class FPNDetector(nn.Module):
    """ResNet-50 + FPN + RPN + Fast R-CNN head (추론 경로 + 학습에 필요한 부품 노출)."""

    STRIDES = (4, 8, 16, 32, 64)
    SIZES = (32, 64, 128, 256, 512)

    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50Backbone()
        self.fpn = FPN()
        self.rpn_head = RPNHead(256, 3)
        self.roi_pool = MultiLevelRoIAlign()
        self.box_head = BoxHead(256, 7, num_classes)

    def features(self, images):
        return self.fpn(self.backbone(images))  # [P2..P6]

    def rpn(self, feats, image_hw):
        """objectness (N,A_tot), deltas (N,A_tot,4), anchors (A_tot,4), 레벨별 앵커 수."""
        obj, reg = self.rpn_head(feats)
        anchors, counts = generate_anchors([f.shape[-2:] for f in feats], self.STRIDES, self.SIZES, device=obj.device)
        return obj, reg, anchors, counts

    @torch.no_grad()
    def propose(self, obj, reg, anchors, counts, image_hw, pre_nms_top_n=1000, post_nms_top_n=1000, nms_thr=0.7):
        """레벨별 top-k -> 디코딩 -> 클립 -> NMS -> 상위 post_nms_top_n. 이미지별 리스트."""
        h, w = image_hw
        proposals = []
        for i in range(obj.size(0)):
            boxes, scores, start = [], [], 0
            for c in counts:
                s = obj[i, start : start + c]
                k = min(pre_nms_top_n, c)
                top = s.topk(k).indices
                boxes.append(decode_boxes(reg[i, start : start + c][top], anchors[start : start + c][top]))
                scores.append(s[top])
                start += c
            b, s = torch.cat(boxes), torch.cat(scores)
            b[:, 0::2] = b[:, 0::2].clamp(0, w)
            b[:, 1::2] = b[:, 1::2].clamp(0, h)
            ok = ((b[:, 2] - b[:, 0]) >= 1.0) & ((b[:, 3] - b[:, 1]) >= 1.0)
            b, s = b[ok], s[ok]
            keep = nms(b, s, nms_thr)[:post_nms_top_n]
            proposals.append(b[keep])
        return proposals

    def box_forward(self, feats, rois):
        """feats: [P2..P6] 전체를 받아 P2~P5 만 사용."""
        pooled = self.roi_pool(feats[:4], rois)
        return self.box_head(pooled)

    @torch.no_grad()
    def forward(self, images):
        """추론: (proposals, cls_logits, box_deltas) 반환."""
        h, w = images.shape[-2:]
        feats = self.features(images)
        obj, reg, anchors, counts = self.rpn(feats, (h, w))
        props = self.propose(obj, reg, anchors, counts, (h, w), pre_nms_top_n=200, post_nms_top_n=100)
        logits, deltas = self.box_forward(feats, props)
        return props, logits, deltas


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FPNDetector(num_classes=21).eval()
    x = torch.randn(2, 3, 256, 320)

    with torch.no_grad():
        feats = model.features(x)
        for name, f in zip(["P2", "P3", "P4", "P5", "P6"], feats):
            print(f"{name}: {tuple(f.shape)}")
        assert [f.shape[1] for f in feats] == [256] * 5
        assert feats[0].shape[-2:] == (64, 80) and feats[4].shape[-2:] == (4, 5)

        obj, reg, anchors, counts = model.rpn(feats, (256, 320))
        print("anchors per level:", counts, "| total:", anchors.shape[0])
        print("objectness:", tuple(obj.shape), "| rpn deltas:", tuple(reg.shape))
        assert obj.shape == (2, anchors.shape[0]) and reg.shape == (2, anchors.shape[0], 4)

        # RoI 레벨 할당 확인: 큰 박스일수록 깊은 레벨(P5)
        test = torch.tensor([[0, 0, 32, 32], [0, 0, 112, 112], [0, 0, 224, 224], [0, 0, 448, 448]], dtype=torch.float32)
        print("RoI size -> level index (0=P2):", map_rois_to_levels(test).tolist())

        props, logits, deltas = model(x)
        print("proposals:", [tuple(p.shape) for p in props])
        print("box head cls:", tuple(logits.shape), "| box deltas:", tuple(deltas.shape))
        assert logits.shape == (sum(len(p) for p in props), 21)
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
