"""
Mask R-CNN (He, Gkioxari, Dollár, Girshick, 2017) — from-scratch PyTorch implementation.

구성 (Faster R-CNN + FPN + mask branch)
  - 백본: ResNet-50 (C2~C5, stride 4/8/16/32) + FPN (P2~P5, 그리고 P6 = P5 의 stride-2 max pool; RPN 전용)
  - RPN: 3x3 conv(256)+ReLU -> objectness(A) / box delta(4A). 레벨마다 앵커 면적 32^2~512^2 (P2~P6),
         종횡비 {1:2, 1:1, 2:1} -> A=3. 모든 레벨이 head 가중치 공유
  - RoIAlign: 소수점 좌표를 반올림(quantize)하지 않고 bilinear 보간으로 특징을 추출.
         torchvision.ops.roi_align(aligned=True) 사용. RoI -> FPN 레벨 할당 k = floor(4 + log2(sqrt(wh)/224)), [2,5] 로 clamp
  - Box head: RoIAlign 7x7 -> FC1024 -> FC1024 -> (class score K+1, 클래스별 box delta 4K)
  - Mask head: RoIAlign 14x14 -> 3x3 conv(256)+ReLU x4 -> 2x2 deconv(stride 2, 256)+ReLU -> 1x1 conv(K) = 28x28 마스크
  - 손실: L = L_cls + L_box + L_mask (+ RPN 손실 2개)
      L_mask = 양성 RoI 에 대해 **정답 클래스 채널 하나만** 골라 픽셀별 sigmoid + binary cross-entropy (클래스 간 경쟁 없음: decoupling)

원 논문과의 차이(단순화):
  - 사전학습(ImageNet) 가중치 없음, 백본 BatchNorm 은 일반 BN (논문/공식 구현은 frozen BN). 배치가 아주 작으면 BN 통계가 불안정
  - RPN 학습 시 이미지 경계를 벗어나는 앵커를 무시하지 않음. NMS/IoU/RoIAlign 은 torchvision.ops 의 batched_nms, roi_align 사용
    (박스 인코딩/디코딩, 앵커, 매칭, 샘플링, FPN, 모든 head, 손실은 직접 구현)
  - 입력은 같은 크기의 이미지 배치 (N,3,H,W) (논문은 짧은 변 800 으로 리사이즈 후 배치 내 패딩)
  - 마스크를 원본 크기에 붙일 때(paste) 박스를 정수 픽셀로 올림/내림하여 bilinear 리사이즈 (공식 구현은 더 정밀한 방식)
  - num_classes 는 배경 포함 총 개수 (COCO 라면 80 + 1 = 81). 라벨 0 = 배경, 1..K = 전경 클래스

좌표 규약: 픽셀 단위 (xmin, ymin, xmax, ymax).
targets: 이미지마다 dict(boxes=(G,4) float, labels=(G,) long in 1..K, masks=(G,H,W) uint8/bool)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, roi_align


# ---------------------------------------------------------------- box utils
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode_boxes(gt: torch.Tensor, ref: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """(tx, ty, tw, th) = ((gx-rx)/rw, (gy-ry)/rh, log(gw/rw), log(gh/rh)) * weights"""
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
    cx, cy = rx + deltas[:, 0] / wx * rw, ry + deltas[:, 1] / wy * rh
    w = rw * torch.exp((deltas[:, 2] / ww).clamp(max=clip))
    h = rh * torch.exp((deltas[:, 3] / wh).clamp(max=clip))
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=1)


def clip_boxes(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return torch.stack([boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h), boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], 1)


def sample_indices(labels: torch.Tensor, num: int, pos_fraction: float):
    """labels: 1 = 양성, 0 = 음성, -1 = 무시. 양성 최대 num*pos_fraction 개, 나머지를 음성으로 무작위 샘플링."""
    pos = (labels == 1).nonzero(as_tuple=True)[0]
    neg = (labels == 0).nonzero(as_tuple=True)[0]
    n_pos = min(int(num * pos_fraction), pos.numel())
    n_neg = min(num - n_pos, neg.numel())
    pos = pos[torch.randperm(pos.numel(), device=pos.device)[:n_pos]]
    neg = neg[torch.randperm(neg.numel(), device=neg.device)[:n_neg]]
    return pos, neg


# ---------------------------------------------------------------- ResNet-50 + FPN
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


class FPN(nn.Module):
    """lateral 1x1 + top-down(nearest 2x 업샘플 후 합산) + 3x3 smoothing -> P2~P5, P6 = maxpool(P5, stride 2)"""

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        outs = [s(l) for s, l in zip(self.smooth, lat)]
        outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))  # P6
        return outs  # [P2, P3, P4, P5, P6]


# ---------------------------------------------------------------- RPN
def generate_anchors(feature_shapes, strides, sizes, ratios=(0.5, 1.0, 2.0), device="cpu") -> torch.Tensor:
    """순서: level -> row -> col -> ratio. head 출력의 flatten 순서와 일치. ratio = h / w, 면적 = size^2."""
    out = []
    for (h, w), stride, size in zip(feature_shapes, strides, sizes):
        r = torch.tensor(ratios, dtype=torch.float32, device=device)
        ws, hs = size / torch.sqrt(r), size * torch.sqrt(r)
        base = torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)  # (A,4)
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx, cy, cx, cy], dim=-1).reshape(-1, 1, 4)
        out.append((centers + base[None]).reshape(-1, 4))
    return torch.cat(out)


class RPN(nn.Module):
    STRIDES = (4, 8, 16, 32, 64)
    SIZES = (32, 64, 128, 256, 512)

    def __init__(self, in_channels=256, num_anchors=3, pre_nms_top_n=1000, post_nms_top_n=1000, nms_thr=0.7,
                 batch_size_per_image=256, pos_fraction=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.pre_nms_top_n, self.post_nms_top_n, self.nms_thr = pre_nms_top_n, post_nms_top_n, nms_thr
        self.batch_size, self.pos_fraction = batch_size_per_image, pos_fraction

    def forward(self, feats, image_size, targets: Optional[List[Dict]] = None):
        n = feats[0].size(0)
        obj, reg, counts = [], [], []
        for f in feats:
            t = F.relu(self.conv(f))
            obj.append(self.cls(t).permute(0, 2, 3, 1).reshape(n, -1))  # (N, H*W*A)
            reg.append(self.reg(t).permute(0, 2, 3, 1).reshape(n, -1, 4))  # (N, H*W*A, 4)
            counts.append(obj[-1].size(1))
        anchors = generate_anchors([f.shape[-2:] for f in feats], self.STRIDES, self.SIZES, device=feats[0].device)
        obj, reg = torch.cat(obj, 1), torch.cat(reg, 1)

        proposals = self._proposals(obj.detach(), reg.detach(), anchors, counts, image_size)
        losses = {}
        if targets is not None:
            losses = self._loss(obj, reg, anchors, targets)
        return proposals, losses

    @torch.no_grad()
    def _proposals(self, obj, reg, anchors, counts, image_size) -> List[torch.Tensor]:
        h, w = image_size
        level_ids = torch.cat([torch.full((c,), i, device=obj.device) for i, c in enumerate(counts)])
        out = []
        for i in range(obj.size(0)):
            boxes_l, scores_l, lv_l = [], [], []
            start = 0
            for lv, c in enumerate(counts):  # 레벨별로 objectness 상위 k 개만 NMS 후보로 (논문/공식 구현과 동일)
                sc = obj[i, start : start + c]
                k = min(self.pre_nms_top_n, c)
                top = sc.topk(k).indices + start
                boxes_l.append(decode_boxes(reg[i, top], anchors[top]))
                scores_l.append(obj[i, top])
                lv_l.append(level_ids[top])
                start += c
            boxes, scores, lv = torch.cat(boxes_l), torch.cat(scores_l), torch.cat(lv_l)
            boxes = clip_boxes(boxes, h, w)
            keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            boxes, scores, lv = boxes[keep], scores[keep], lv[keep]
            keep = batched_nms(boxes, scores, lv, self.nms_thr)[: self.post_nms_top_n]  # 레벨별 NMS
            out.append(boxes[keep])
        return out

    def _loss(self, obj, reg, anchors, targets):
        cls_loss, box_loss, total = obj.new_zeros(()), obj.new_zeros(()), 0
        for i, t in enumerate(targets):
            gt = t["boxes"]
            labels = torch.full((anchors.size(0),), -1, dtype=torch.long, device=anchors.device)
            if gt.numel() == 0:
                labels[:] = 0
                matched = torch.zeros(anchors.size(0), 4, device=anchors.device)
            else:
                iou = box_iou(gt, anchors)  # (G, A_tot)
                max_iou, idx = iou.max(0)
                labels[max_iou < 0.3] = 0
                labels[max_iou >= 0.7] = 1
                best = iou.max(1).values  # 각 GT 와 IoU 가 가장 큰 앵커(동점 포함)도 양성
                labels[((iou == best[:, None]) & (best[:, None] > 0)).any(0)] = 1
                matched = gt[idx]
            pos, neg = sample_indices(labels, self.batch_size, self.pos_fraction)
            sel = torch.cat([pos, neg])
            tgt = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).float()
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(obj[i, sel], tgt, reduction="sum")
            if pos.numel():
                tb = encode_boxes(matched[pos], anchors[pos])
                box_loss = box_loss + F.smooth_l1_loss(reg[i, pos], tb, beta=1 / 9, reduction="sum")
            total += sel.numel()
        total = max(total, 1)
        return {"rpn_objectness": cls_loss / total, "rpn_box": box_loss / total}


# ---------------------------------------------------------------- RoIAlign (multi-scale)
class MultiScaleRoIAlign(nn.Module):
    """RoI 크기에 따라 FPN 레벨을 고르고 RoIAlign: k = floor(4 + log2(sqrt(wh)/224)), clamp [2,5]."""

    def __init__(self, strides=(4, 8, 16, 32), sampling_ratio: int = 2):
        super().__init__()
        self.strides, self.sampling_ratio = strides, sampling_ratio

    def forward(self, feats: List[torch.Tensor], rois: List[torch.Tensor], output_size: int) -> torch.Tensor:
        all_rois = torch.cat([torch.cat([torch.full_like(r[:, :1], i), r], 1) for i, r in enumerate(rois)], 0)
        wh = (all_rois[:, 3] - all_rois[:, 1]).clamp(min=1e-6) * (all_rois[:, 4] - all_rois[:, 2]).clamp(min=1e-6)
        lv = torch.floor(4 + torch.log2(wh.sqrt() / 224 + 1e-8)).clamp(2, 5).long() - 2
        out = feats[0].new_zeros(all_rois.size(0), feats[0].size(1), output_size, output_size)
        for l, (f, s) in enumerate(zip(feats, self.strides)):
            idx = (lv == l).nonzero(as_tuple=True)[0]
            if idx.numel():
                # aligned=True: 픽셀 중심 보정(-0.5)을 포함한 RoIAlign. 좌표를 반올림하지 않고 bilinear 보간
                out[idx] = roi_align(f, all_rois[idx], output_size, spatial_scale=1.0 / s,
                                     sampling_ratio=self.sampling_ratio, aligned=True)
        return out


# ---------------------------------------------------------------- heads
class BoxHead(nn.Module):
    """RoI 7x7 -> FC1024 -> FC1024 -> (class logits K+1, class-specific box deltas 4*(K+1))"""

    def __init__(self, in_channels=256, pool=7, hidden=1024, num_classes=81):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * pool * pool, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, num_classes)
        self.reg = nn.Linear(hidden, num_classes * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.reg.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.reg.bias)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))
        return self.cls(x), self.reg(x)


class MaskHead(nn.Module):
    """RoI 14x14 -> conv3x3 x4 -> deconv 2x2 (stride 2) -> 28x28 -> 1x1 conv (K 채널, 배경 제외)"""

    def __init__(self, in_channels=256, num_fg_classes=80, layers=4):
        super().__init__()
        convs = []
        for _ in range(layers):
            convs += [nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True)]
            in_channels = 256
        self.convs = nn.Sequential(*convs)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.predictor = nn.Conv2d(256, num_fg_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.predictor(F.relu(self.deconv(self.convs(x))))  # (R, K, 28, 28) logits


def paste_mask(mask: torch.Tensor, box: torch.Tensor, h: int, w: int, thr: float = 0.5) -> torch.Tensor:
    """28x28 확률 마스크를 박스 크기로 bilinear 리사이즈해서 (h, w) 캔버스에 붙이고 thr 로 이진화."""
    x0, y0 = max(int(box[0].floor()), 0), max(int(box[1].floor()), 0)
    x1, y1 = min(int(box[2].ceil()), w), min(int(box[3].ceil()), h)
    canvas = torch.zeros(h, w, dtype=torch.bool, device=mask.device)
    if x1 > x0 and y1 > y0:
        m = F.interpolate(mask[None, None], size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0]
        canvas[y0:y1, x0:x1] = m >= thr
    return canvas


# ---------------------------------------------------------------- Mask R-CNN
class MaskRCNN(nn.Module):
    BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

    def __init__(self, num_classes: int = 81, rpn_pre_nms_top_n=1000, rpn_post_nms_top_n=1000,
                 box_batch_size=512, box_pos_fraction=0.25, score_thr=0.05, nms_thr=0.5, max_det=100,
                 mask_size=28):
        super().__init__()
        self.num_classes = num_classes  # 배경 포함
        self.backbone, self.fpn = ResNet50Backbone(), FPN()
        self.rpn = RPN(pre_nms_top_n=rpn_pre_nms_top_n, post_nms_top_n=rpn_post_nms_top_n)
        self.pooler = MultiScaleRoIAlign()
        self.box_head = BoxHead(num_classes=num_classes)
        self.mask_head = MaskHead(num_fg_classes=num_classes - 1)
        self.box_batch_size, self.box_pos_fraction = box_batch_size, box_pos_fraction
        self.score_thr, self.nms_thr, self.max_det, self.mask_size = score_thr, nms_thr, max_det, mask_size

    # ---- training: RoI 샘플링 (양성 IoU>=0.5, 이미지당 512 개, 양성 25%)
    @torch.no_grad()
    def _select_samples(self, proposals, targets):
        out_boxes, out_labels, out_match, out_gt = [], [], [], []
        for props, t in zip(proposals, targets):
            gt, gl = t["boxes"], t["labels"]
            props = torch.cat([props, gt], 0)  # GT 박스도 제안에 추가 (초기 학습 안정화)
            if gt.numel() == 0:
                labels = torch.zeros(props.size(0), dtype=torch.long, device=props.device)
                match = torch.zeros_like(labels)
            else:
                max_iou, match = box_iou(gt, props).max(0)
                labels = gl[match].clone()
                labels[max_iou < 0.5] = 0  # 배경
            flag = (labels > 0).long()  # sample_indices 는 1=양성, 0=음성
            pos, neg = sample_indices(flag, self.box_batch_size, self.box_pos_fraction)
            keep = torch.cat([pos, neg])
            out_boxes.append(props[keep])
            out_labels.append(labels[keep])
            out_match.append(match[keep])
            out_gt.append(gt[match[keep]] if gt.numel() else props[keep])  # 회귀 타깃용 매칭 GT 박스
        return out_boxes, out_labels, out_match, out_gt

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None):
        """학습(targets 제공): loss dict. 추론: 이미지별 dict(boxes, labels, scores, masks(bool, H x W))."""
        if self.training and targets is None:
            raise ValueError("학습 모드에서는 targets 가 필요합니다")
        h, w = images.shape[-2:]
        feats = self.fpn(self.backbone(images))
        proposals, losses = self.rpn(feats, (h, w), targets if self.training else None)
        roi_feats = feats[:4]  # P2~P5 만 RoIAlign 에 사용 (P6 은 RPN 전용)

        if self.training:
            boxes, labels, match, gt_boxes = self._select_samples(proposals, targets)
            losses.update(self._box_loss(roi_feats, boxes, labels, gt_boxes))
            losses.update(self._mask_loss(roi_feats, boxes, labels, match, targets))
            return losses
        return self._inference(roi_feats, proposals, h, w)

    def _box_loss(self, feats, boxes, labels, gt_boxes):
        x = self.pooler(feats, boxes, 7)
        logits, deltas = self.box_head(x)
        labels_all = torch.cat(labels)
        cls_loss = F.cross_entropy(logits, labels_all)
        pos = (labels_all > 0).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            return {"box_cls": cls_loss, "box_reg": deltas.sum() * 0}
        # 정답 클래스의 box delta 만 사용 (class-specific regression), 양성 RoI 에서만 smooth L1
        d = deltas.view(deltas.size(0), self.num_classes, 4)[pos, labels_all[pos]]
        tgt = encode_boxes(torch.cat(gt_boxes)[pos], torch.cat(boxes)[pos], self.BOX_WEIGHTS)
        return {"box_cls": cls_loss, "box_reg": F.smooth_l1_loss(d, tgt, beta=1.0, reduction="sum") / labels_all.numel()}

    def _mask_loss(self, feats, boxes, labels, match, targets):
        pos_boxes, pos_labels, pos_masks = [], [], []
        for b, l, m, t in zip(boxes, labels, match, targets):
            p = l > 0
            pos_boxes.append(b[p])
            pos_labels.append(l[p])
            pos_masks.append(t["masks"][m[p]] if p.any() else t["masks"][:0])
        n_pos = sum(b.size(0) for b in pos_boxes)
        if n_pos == 0:
            return {"mask": sum(p.sum() for p in self.mask_head.parameters()) * 0}
        x = self.pooler(feats, pos_boxes, 14)
        logits = self.mask_head(x)  # (M, K, 28, 28)
        lab = torch.cat(pos_labels)
        logits = logits[torch.arange(n_pos, device=logits.device), lab - 1]  # 정답 클래스 채널 하나만 (클래스 간 경쟁 없음)
        # 정답 마스크를 RoI 영역으로 RoIAlign 하여 28x28 타깃 생성 (예측과 같은 좌표계)
        gt_masks = torch.cat(pos_masks).float().unsqueeze(1)  # (M, 1, H, W)
        rois = torch.cat([torch.arange(n_pos, device=gt_masks.device, dtype=torch.float32)[:, None], torch.cat(pos_boxes)], 1)
        target = roi_align(gt_masks, rois, self.mask_size, spatial_scale=1.0, sampling_ratio=2, aligned=True)[:, 0]
        target = (target >= 0.5).float()
        return {"mask": F.binary_cross_entropy_with_logits(logits, target)}

    @torch.no_grad()
    def _inference(self, feats, proposals, h, w):
        x = self.pooler(feats, proposals, 7)
        logits, deltas = self.box_head(x)
        counts = [p.size(0) for p in proposals]
        det_boxes, det_scores, det_labels = [], [], []
        for p, lg, dl in zip(proposals, logits.split(counts), deltas.split(counts)):
            r = p.size(0)
            scores = F.softmax(lg, -1)[:, 1:]  # (R, K) 배경 제외
            boxes = decode_boxes(dl.view(-1, 4), p[:, None].expand(r, self.num_classes, 4).reshape(-1, 4), self.BOX_WEIGHTS)
            boxes = clip_boxes(boxes, h, w).view(r, self.num_classes, 4)[:, 1:]  # (R, K, 4)
            labels = torch.arange(1, self.num_classes, device=p.device)[None].expand(r, -1)
            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            keep = (scores > self.score_thr) & ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = batched_nms(boxes, scores, labels, self.nms_thr)[: self.max_det]
            det_boxes.append(boxes[keep]); det_scores.append(scores[keep]); det_labels.append(labels[keep])

        # 마스크 분기는 최종 검출 박스에 대해서만 실행 (논문: 추론 시 상위 100 검출에 대해 마스크 예측 -> 병렬이라 오버헤드 작음)
        results = []
        if sum(b.size(0) for b in det_boxes) > 0:
            probs = self.mask_head(self.pooler(feats, det_boxes, 14)).sigmoid()
        k = 0
        for b, s, l in zip(det_boxes, det_scores, det_labels):
            n = b.size(0)
            if n:
                m = probs[k : k + n][torch.arange(n), l - 1]  # 예측 클래스 채널의 28x28 확률
                masks = torch.stack([paste_mask(m[j], b[j], h, w) for j in range(n)])
            else:
                masks = torch.zeros(0, h, w, dtype=torch.bool, device=b.device)
            k += n
            results.append({"boxes": b, "scores": s, "labels": l, "masks": masks})
        return results


if __name__ == "__main__":
    torch.manual_seed(0)
    K = 4  # 전경 클래스 수 (+ 배경 = 5)
    model = MaskRCNN(num_classes=K + 1, rpn_pre_nms_top_n=300, rpn_post_nms_top_n=100, box_batch_size=64)
    n, H, W = 2, 256, 320
    images = torch.randn(n, 3, H, W)

    # 더미 정답: 이미지마다 박스 2개 + 박스 내부를 채운 사각형 마스크
    targets = []
    for _ in range(n):
        boxes = torch.tensor([[30.0, 40.0, 130.0, 150.0], [160.0, 60.0, 290.0, 220.0]])
        masks = torch.zeros(2, H, W, dtype=torch.uint8)
        for j, b in enumerate(boxes.long()):
            masks[j, b[1] : b[3], b[0] : b[2]] = 1
        targets.append({"boxes": boxes, "labels": torch.tensor([1, 3]), "masks": masks})

    # 1) FPN / 앵커 shape
    with torch.no_grad():
        feats = model.fpn(model.backbone(images))
    print("FPN:", {f"P{i + 2}": tuple(f.shape[-2:]) for i, f in enumerate(feats)})

    # 2) 학습 모드: loss 계산 + backward
    model.train()
    losses = model(images, targets)
    print("losses:", {k: round(v.item(), 4) for k, v in losses.items()})
    total = sum(losses.values())
    total.backward()
    assert set(losses) == {"rpn_objectness", "rpn_box", "box_cls", "box_reg", "mask"}
    assert all(torch.isfinite(v) for v in losses.values())

    # 3) RoIAlign / 헤드 shape 확인
    model.eval()
    with torch.no_grad():
        rois = [torch.tensor([[10.0, 10.0, 40.0, 40.0], [0.0, 0.0, 250.0, 250.0]]) for _ in range(n)]
        r7 = model.pooler(feats[:4], rois, 7)
        r14 = model.pooler(feats[:4], rois, 14)
        print("RoIAlign 7x7:", tuple(r7.shape), "| 14x14:", tuple(r14.shape))
        print("box head:", [tuple(t.shape) for t in model.box_head(r7)], "| mask head:", tuple(model.mask_head(r14).shape))
        assert r7.shape == (4, 256, 7, 7) and model.mask_head(r14).shape == (4, K, 28, 28)

        # 4) 추론: 이미지별 boxes / scores / labels / masks
        model.score_thr = 0.0  # 무작위 초기화라 점수가 낮으므로 임계값을 낮춰 출력 형태만 확인
        dets = model(images)
    for i, d in enumerate(dets):
        print(f"image {i}: boxes {tuple(d['boxes'].shape)}, scores {tuple(d['scores'].shape)}, "
              f"labels {tuple(d['labels'].shape)}, masks {tuple(d['masks'].shape)} {d['masks'].dtype}")
        assert d["masks"].shape == (d["boxes"].size(0), H, W)
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
