"""
RetinaNet (Lin et al., 2017) — from-scratch PyTorch implementation.

구성
  - ResNet-50 백본 (C3~C5, stride 8/16/32)
  - FPN: P3~P5 (lateral 1x1 + top-down + 3x3 smoothing), P6 = C5 위의 3x3 stride-2 conv, P7 = ReLU 후 3x3 stride-2 conv
  - 앵커: 레벨마다 면적 32^2 ~ 512^2 (P3~P7), 종횡비 {1:2, 1:1, 2:1} x 스케일 {2^0, 2^(1/3), 2^(2/3)} = 9개
  - 분류/회귀 서브넷: 각각 3x3 conv(256)+ReLU 4번 -> 최종 conv (K*A / 4*A). 레벨 간 가중치 공유, 두 서브넷은 서로 별개
  - 분류 출력 bias 를 prior pi=0.01 로 초기화 (b = -log((1-pi)/pi))
  - Focal Loss: FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t), alpha=0.25, gamma=2

원 논문과의 차이(단순화):
  - 백본의 BatchNorm 은 일반 BN (논문은 학습 중 고정(frozen) BN), 사전학습 가중치 없음(무작위 초기화)
  - 서브넷에 정규화 층 없음 (논문과 동일)
  - 이미지 크기는 128 의 배수를 가정 (P7 까지 stride 128 이 정수로 나누어 떨어지도록)
  - NMS 는 torchvision.ops.batched_nms 사용. 그 외 박스 연산(IoU/encode/decode/앵커/focal loss)은 직접 구현

좌표 규약: 픽셀 단위 (xmin, ymin, xmax, ymax).
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


# ---------------------------------------------------------------- box utils
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode_boxes(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([(gx - rx) / rw, (gy - ry) / rh, torch.log(gw / rw), torch.log(gh / rh)], dim=1)


def decode_boxes(deltas: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    rw, rh = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    rx, ry = ref[:, 0] + 0.5 * rw, ref[:, 1] + 0.5 * rh
    clip = math.log(1000.0 / 16)
    cx, cy = rx + deltas[:, 0] * rw, ry + deltas[:, 1] * rh
    w, h = rw * torch.exp(deltas[:, 2].clamp(max=clip)), rh * torch.exp(deltas[:, 3].clamp(max=clip))
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=1)


# ---------------------------------------------------------------- Focal Loss
def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "none") -> torch.Tensor:
    """
    logits, targets: 같은 shape. targets 는 0/1 (one-hot). 클래스마다 독립적인 이진 문제로 취급.
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)   (gamma=0, alpha_t=1 이면 BCE)
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log(p_t)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


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
    """이미지 -> [C3, C4, C5] (채널 512/1024/2048, stride 8/16/32)"""

    out_channels = (512, 1024, 2048)

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
        return [c3, c4, c5]  # C2 는 메모리 문제로 사용하지 않음


# ---------------------------------------------------------------- FPN (P3~P7)
class FPNP3P7(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])
        self.p6 = nn.Conv2d(in_channels[-1], out_channels, 3, stride=2, padding=1)  # 논문: C5 에서 계산
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # ReLU(P6) 에서 계산

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        c5 = feats[-1]
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        p3, p4, p5 = [s(l) for s, l in zip(self.smooth, lat)]
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p3, p4, p5, p6, p7]


# ---------------------------------------------------------------- anchors
def generate_anchors(feature_shapes, strides, base_sizes=(32, 64, 128, 256, 512),
                     ratios=(0.5, 1.0, 2.0), scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)), device="cpu") -> torch.Tensor:
    """(총 앵커 수, 4). 순서: level -> row -> col -> (ratio, scale). head 출력 flatten 순서와 일치. ratio = h / w."""
    out = []
    for (h, w), stride, size in zip(feature_shapes, strides, base_sizes):
        r = torch.tensor(ratios, dtype=torch.float32, device=device)[:, None].expand(-1, len(scales)).reshape(-1)
        s = torch.tensor(scales, dtype=torch.float32, device=device)[None, :].expand(len(ratios), -1).reshape(-1)
        side = size * s  # 면적 = side^2
        ws, hs = side / torch.sqrt(r), side * torch.sqrt(r)
        base = torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)  # (9,4)
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 1, 4)
        out.append((centers + base[None]).reshape(-1, 4))
    return torch.cat(out)


# ---------------------------------------------------------------- heads
def _subnet(c: int) -> nn.Sequential:
    layers = []
    for _ in range(4):
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class RetinaHead(nn.Module):
    """class subnet(K*A) + box subnet(4*A). 모든 피라미드 레벨에서 가중치 공유."""

    def __init__(self, in_channels=256, num_anchors=9, num_classes=80, prior=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tower, self.box_tower = _subnet(in_channels), _subnet(in_channels)
        self.cls_out = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.box_out = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # 핵심 초기화: 학습 시작 시 모든 앵커의 foreground 확률이 pi 가 되도록
        # -> 압도적인 negative 에 의한 초기 불안정 방지
        nn.init.constant_(self.cls_out.bias, -math.log((1 - prior) / prior))

    def forward(self, feats):
        cls, box = [], []
        for f in feats:
            n = f.size(0)
            cls.append(self.cls_out(self.cls_tower(f)).permute(0, 2, 3, 1).reshape(n, -1, self.num_classes))
            box.append(self.box_out(self.box_tower(f)).permute(0, 2, 3, 1).reshape(n, -1, 4))
        return torch.cat(cls, dim=1), torch.cat(box, dim=1)  # (N, A_tot, K), (N, A_tot, 4)


class RetinaNet(nn.Module):
    STRIDES = (8, 16, 32, 64, 128)
    SIZES = (32, 64, 128, 256, 512)

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50Backbone()
        self.fpn = FPNP3P7()
        self.head = RetinaHead(256, 9, num_classes)

    def forward(self, images):
        """반환: cls_logits (N,A,K), box_deltas (N,A,4), anchors (A,4), P3~P7 feature 리스트"""
        feats = self.fpn(self.backbone(images))
        cls, box = self.head(feats)
        anchors = generate_anchors([f.shape[-2:] for f in feats], self.STRIDES, self.SIZES, device=images.device)
        return cls, box, anchors, feats

    @torch.no_grad()
    def detect(self, images, score_thr=0.05, top_k=1000, nms_thr=0.5, max_det=100):
        """추론: 점수 > 0.05 인 (앵커, 클래스) 쌍 중 상위 1000개 -> 디코딩 -> 클래스별 NMS(0.5)."""
        cls, box, anchors, _ = self.forward(images)
        h, w = images.shape[-2:]
        results = []
        for i in range(images.size(0)):
            scores = cls[i].sigmoid().flatten()
            cand = (scores > score_thr).nonzero(as_tuple=True)[0]
            sc = scores[cand]
            top = sc.topk(min(top_k, sc.numel())).indices
            cand, sc = cand[top], sc[top]
            a_idx, labels = cand // self.num_classes, cand % self.num_classes
            boxes = decode_boxes(box[i][a_idx], anchors[a_idx])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            k = batched_nms(boxes, sc, labels, nms_thr)[:max_det]
            results.append((boxes[k], sc[k], labels[k]))
        return results


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RetinaNet(num_classes=20).eval()
    x = torch.randn(2, 3, 256, 384)

    with torch.no_grad():
        cls, box, anchors, feats = model(x)
        for name, f in zip(["P3", "P4", "P5", "P6", "P7"], feats):
            print(f"{name}: {tuple(f.shape)}")
        print("anchors:", tuple(anchors.shape), "| cls logits:", tuple(cls.shape), "| box deltas:", tuple(box.shape))
        n_anchor = sum(f.shape[-2] * f.shape[-1] for f in feats) * 9
        assert anchors.shape == (n_anchor, 4) and cls.shape == (2, n_anchor, 20) and box.shape == (2, n_anchor, 4)
        assert feats[0].shape[-2:] == (32, 48) and feats[4].shape[-2:] == (2, 3)

        # prior 초기화 확인: 초기 foreground 확률 ~ 0.01
        print("initial mean fg prob: %.4f" % cls.sigmoid().mean().item())

        # Focal loss 동작 확인: 쉬운 샘플(p_t 큼)의 손실이 BCE 대비 크게 줄어드는지
        logit = torch.tensor([4.0, 0.0, -4.0])  # target=1 일 때 쉬움/모호/어려움
        tgt = torch.ones(3)
        bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        fl = sigmoid_focal_loss(logit, tgt)
        print("BCE :", [round(v, 4) for v in bce.tolist()])
        print("FL  :", [round(v, 4) for v in fl.tolist()], "(쉬운 샘플은 거의 0으로 down-weight)")

        dets = model.detect(x)
        print("detections per image:", [tuple(d[0].shape) for d in dets])
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
