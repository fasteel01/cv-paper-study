"""
Fast R-CNN (Girshick, 2015) — from-scratch PyTorch implementation.

구성요소: RoIPool(순수 PyTorch), conv 백본 + FC6/FC7 + 분류/회귀 형제 헤드,
multi-task loss(cross-entropy + smooth L1), RoI 샘플링, Truncated SVD FC 압축.

원 논문과의 차이점(현대적 단순화):
  - ImageNet 사전학습 VGG-16/CaffeNet 대신 작은 conv 백본(feature stride 16)을 처음부터 구성한다.
  - Selective Search는 구현하지 않고 제안 박스를 외부 입력(더미)으로 받는다.
  - RoIPool은 논문처럼 좌표를 stride로 나눠 반올림(양자화)한다(RoIAlign 아님).
  - 박스 타깃 정규화(평균/표준편차)는 `box_std`로 단순화(평균 0 가정).
  - 박스 좌표는 (x1, y1, x2, y2) 픽셀 단위, RoI는 (batch_idx, x1, y1, x2, y2) 형식.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 박스 유틸리티
# ----------------------------------------------------------------------------
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def encode_boxes(proposals: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(tx, ty, tw, th) 회귀 타깃."""
    pw, ph = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    px, py = proposals[:, 0] + 0.5 * pw, proposals[:, 1] + 0.5 * ph
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([(gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)], dim=1)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    pw, ph = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    px, py = proposals[:, 0] + 0.5 * pw, proposals[:, 1] + 0.5 * ph
    gx, gy = pw * deltas[:, 0] + px, ph * deltas[:, 1] + py
    gw, gh = pw * torch.exp(deltas[:, 2]), ph * torch.exp(deltas[:, 3])
    return torch.stack([gx - 0.5 * gw, gy - 0.5 * gh, gx + 0.5 * gw, gy + 0.5 * gh], dim=1)


# ----------------------------------------------------------------------------
# RoI Pooling
# ----------------------------------------------------------------------------
def roi_pool(features: torch.Tensor, rois: torch.Tensor, output_size: Tuple[int, int], stride: int) -> torch.Tensor:
    """
    features: (N, C, Hf, Wf), rois: (R, 5) = (batch_idx, x1, y1, x2, y2) [입력 이미지 좌표]
    반환: (R, C, out_h, out_w)

    1) 좌표를 stride로 나눠 반올림해 특징 맵 좌표로 변환
    2) RoI를 out_h x out_w 격자로 나누고 (셀 크기 ~ h/out_h x w/out_w)
    3) 각 셀에서 max pooling (빈 셀이 생기지 않도록 floor/ceil로 경계 설정)
    autograd가 max의 argmax 위치로 그래디언트를 전달한다.
    """
    out_h, out_w = output_size
    _, C, Hf, Wf = features.shape
    outputs = features.new_zeros(rois.shape[0], C, out_h, out_w)

    for r in range(rois.shape[0]):
        b = int(rois[r, 0].item())
        x1, y1, x2, y2 = [int(round(v.item() / stride)) for v in rois[r, 1:]]
        x1, y1 = max(min(x1, Wf - 1), 0), max(min(y1, Hf - 1), 0)
        x2, y2 = max(min(x2, Wf - 1), x1), max(min(y2, Hf - 1), y1)
        roi_w, roi_h = x2 - x1 + 1, y2 - y1 + 1  # 최소 1x1
        bin_w, bin_h = roi_w / out_w, roi_h / out_h

        for i in range(out_h):
            ys = y1 + int(torch.floor(torch.tensor(i * bin_h)).item())
            ye = y1 + int(torch.ceil(torch.tensor((i + 1) * bin_h)).item())
            ys, ye = min(ys, Hf - 1), max(min(ye, Hf), min(ys, Hf - 1) + 1)
            for j in range(out_w):
                xs = x1 + int(torch.floor(torch.tensor(j * bin_w)).item())
                xe = x1 + int(torch.ceil(torch.tensor((j + 1) * bin_w)).item())
                xs, xe = min(xs, Wf - 1), max(min(xe, Wf), min(xs, Wf - 1) + 1)
                outputs[r, :, i, j] = features[b, :, ys:ye, xs:xe].amax(dim=(1, 2))
    return outputs


class RoIPool(nn.Module):
    def __init__(self, output_size: Tuple[int, int] = (7, 7), stride: int = 16):
        super().__init__()
        self.output_size, self.stride = output_size, stride

    def forward(self, features, rois):
        return roi_pool(features, rois, self.output_size, self.stride)


# ----------------------------------------------------------------------------
# Truncated SVD
# ----------------------------------------------------------------------------
def truncate_linear(fc: nn.Linear, t: int) -> nn.Sequential:
    """W (out x in) ~= U_t S_t V_t^T 로 근사해 Linear(in->t, bias 없음) + Linear(t->out, 원 bias) 로 대체."""
    W = fc.weight.data  # (out, in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    first = nn.Linear(fc.in_features, t, bias=False)
    second = nn.Linear(t, fc.out_features, bias=fc.bias is not None)
    first.weight.data = (torch.diag(S[:t]) @ Vh[:t]).contiguous()
    second.weight.data = U[:, :t].contiguous()
    if fc.bias is not None:
        second.bias.data = fc.bias.data.clone()
    return nn.Sequential(first, second)


# ----------------------------------------------------------------------------
# 네트워크
# ----------------------------------------------------------------------------
def make_backbone() -> nn.Sequential:
    """총 stride 16의 작은 conv 백본 (4번 다운샘플). 출력 채널 256."""

    def block(cin, cout, pool):
        layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return layers

    return nn.Sequential(
        *block(3, 32, True),  # /2
        *block(32, 64, True),  # /4
        *block(64, 128, True),  # /8
        *block(128, 256, True),  # /16
        *block(256, 256, False),
    )


class FastRCNN(nn.Module):
    """
    forward(images, rois):
      images: (N, 3, H, W), rois: (R, 5) (batch_idx, x1, y1, x2, y2)
      -> cls_logits (R, K+1), box_deltas (R, K, 4)
    """

    def __init__(self, num_classes: int = 20, pool_size: int = 7, stride: int = 16, fc_dim: int = 1024):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = make_backbone()
        self.roi_pool = RoIPool((pool_size, pool_size), stride)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pool_size * pool_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        # 두 개의 형제(sibling) 출력층
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)  # 논문의 초기화
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, images: torch.Tensor, rois: torch.Tensor):
        fmap = self.backbone(images)  # 이미지당 conv 1회
        pooled = self.roi_pool(fmap, rois)  # RoI마다 7x7
        h = self.fc(pooled)
        return self.cls_score(h), self.bbox_pred(h).view(-1, self.num_classes, 4)


# ----------------------------------------------------------------------------
# 학습 보조: RoI 샘플링, multi-task loss
# ----------------------------------------------------------------------------
def sample_rois(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    rois_per_image: int = 64,
    pos_fraction: float = 0.25,
    pos_iou: float = 0.5,
    bg_iou_lo: float = 0.1,
):
    """한 이미지에 대해 양성(IoU>=0.5, 최대 25%) / 배경(IoU in [0.1, 0.5)) RoI를 뽑고 라벨·회귀 타깃을 만든다."""
    iou = box_iou(proposals, gt_boxes)
    max_iou, idx = iou.max(dim=1)
    pos = (max_iou >= pos_iou).nonzero().flatten()
    bg = ((max_iou < pos_iou) & (max_iou >= bg_iou_lo)).nonzero().flatten()

    n_pos = min(int(rois_per_image * pos_fraction), len(pos))
    n_bg = min(rois_per_image - n_pos, len(bg))
    pos = pos[torch.randperm(len(pos))[:n_pos]]
    bg = bg[torch.randperm(len(bg))[:n_bg]]
    keep = torch.cat([pos, bg])

    labels = gt_labels[idx[keep]].clone()
    labels[n_pos:] = 0  # 배경
    targets = encode_boxes(proposals[keep], gt_boxes[idx[keep]])
    return proposals[keep], labels, targets


def fast_rcnn_loss(cls_logits, box_deltas, labels, reg_targets, lam: float = 1.0, box_std=(0.1, 0.1, 0.2, 0.2)):
    """L = L_cls + lambda * [u>=1] * L_loc  (smooth L1). 타깃은 box_std로 나누어 정규화."""
    cls_loss = F.cross_entropy(cls_logits, labels)
    pos = labels > 0
    if pos.any():
        std = reg_targets.new_tensor(box_std)
        pred = box_deltas[pos, labels[pos] - 1]
        loc_loss = F.smooth_l1_loss(pred, reg_targets[pos] / std, reduction="sum") / labels.numel()
    else:
        loc_loss = box_deltas.sum() * 0.0
    return cls_loss + lam * loc_loss, cls_loss.detach(), loc_loss.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FastRCNN(num_classes=20)

    images = torch.randn(2, 3, 256, 320)
    rois = torch.tensor(
        [
            [0, 10.0, 20.0, 150.0, 200.0],
            [0, 100.0, 50.0, 300.0, 240.0],
            [1, 0.0, 0.0, 319.0, 255.0],
            [1, 120.0, 100.0, 180.0, 160.0],
        ]
    )
    cls_logits, box_deltas = model(images, rois)
    print("feature map shape:", model.backbone(images).shape)  # [2, 256, 16, 20]
    print("roi pooled shape:", model.roi_pool(model.backbone(images), rois).shape)  # [4, 256, 7, 7]
    print("cls_logits shape:", cls_logits.shape)  # [4, 21]
    print("box_deltas shape:", box_deltas.shape)  # [4, 20, 4]

    # RoIPool 점검: torchvision 구현과 비교하지 않고, 상수 맵에서 값 유지 + max 동작 확인
    f = torch.arange(16.0).view(1, 1, 4, 4)
    out = roi_pool(f, torch.tensor([[0.0, 0.0, 0.0, 48.0, 48.0]]), (2, 2), stride=16)
    print("roi_pool demo (expect [[5,7],[13,15]]):", out[0, 0].tolist())

    # Truncated SVD 점검
    fc = nn.Linear(512, 256)
    approx = truncate_linear(fc, 64)
    x = torch.randn(3, 512)
    print("SVD params:", sum(p.numel() for p in fc.parameters()), "->", sum(p.numel() for p in approx.parameters()))
    print("SVD output shape:", approx(x).shape)
