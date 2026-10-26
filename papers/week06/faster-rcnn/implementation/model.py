"""
Faster R-CNN (Ren et al., 2015) — from-scratch PyTorch implementation.

구성요소: 앵커 생성(3 scale x 3 ratio), RPN(3x3 conv + 1x1 cls/reg), 앵커 타깃 할당 및 256개 샘플링,
제안 생성(박스 디코딩 + 클리핑 + NMS), RoIPool + FC 검출 헤드(Fast R-CNN), RPN/검출 손실.

원 논문과의 차이점(현대적 단순화):
  - ImageNet 사전학습 VGG-16/ZF 대신 작은 conv 백본(feature stride 16)을 처음부터 구성한다.
  - 논문의 4단계 교대 학습(alternating training) 대신 "근사 joint 학습"을 쓴다:
    RPN이 만든 제안은 detach하여(박스 좌표에 대한 그래디언트는 무시) 하나의 손실로 한 번에 역전파한다.
  - RPN 분류는 논문처럼 앵커당 2-way(2k scores)이며 softmax cross-entropy를 쓴다.
    회귀 손실은 정규화 항을 "샘플된 앵커 수"로 통일하고 lambda=1로 둔다(논문은 N_reg~2400, lambda=10으로 사실상 동일 가중치).
  - NMS와 RoIPool을 순수 PyTorch로 구현했다(RoIPool은 RoIAlign이 아닌 좌표 양자화 방식, 속도보다 가독성 우선).
  - 이미지 크기가 같은 배치만 지원한다. 박스 좌표는 (x1, y1, x2, y2) 픽셀 단위.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

BOX_STD = (0.1, 0.1, 0.2, 0.2)  # 검출 헤드 회귀 타깃 정규화


# ----------------------------------------------------------------------------
# 박스 유틸리티 (R-CNN / Fast R-CNN과 동일한 파라미터화)
# ----------------------------------------------------------------------------
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def encode_boxes(ref: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(tx, ty, tw, th): 중심 오프셋은 ref 크기로 정규화, 크기는 log 비율."""
    pw, ph = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    px, py = ref[:, 0] + 0.5 * pw, ref[:, 1] + 0.5 * ph
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([(gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)], dim=1)


def decode_boxes(ref: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    pw, ph = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    px, py = ref[:, 0] + 0.5 * pw, ref[:, 1] + 0.5 * ph
    dw, dh = deltas[:, 2].clamp(max=4.0), deltas[:, 3].clamp(max=4.0)  # exp 폭주 방지
    gx, gy = pw * deltas[:, 0] + px, ph * deltas[:, 1] + py
    gw, gh = pw * torch.exp(dw), ph * torch.exp(dh)
    return torch.stack([gx - 0.5 * gw, gy - 0.5 * gh, gx + 0.5 * gw, gy + 0.5 * gh], dim=1)


def clip_boxes(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return torch.stack(
        [boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h), boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], dim=1
    )


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """점수 내림차순으로 보면서 IoU가 임계값을 넘는 박스를 제거. 살아남은 인덱스 반환."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thresh]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long)


# ----------------------------------------------------------------------------
# 앵커
# ----------------------------------------------------------------------------
def generate_anchors(
    feat_h: int, feat_w: int, stride: int = 16, scales=(128, 256, 512), ratios=(0.5, 1.0, 2.0), device="cpu"
) -> torch.Tensor:
    """
    특징 맵의 모든 위치에 k = len(scales)*len(ratios) 개의 앵커를 둔다. 반환: (H*W*k, 4)
    ratio = h / w. 면적은 scale^2로 유지: w = scale / sqrt(ratio), h = scale * sqrt(ratio).
    순서는 (y, x, anchor)로 RPN 출력 reshape와 일치한다.
    """
    base = []
    for s in scales:
        for r in ratios:
            w, h = s / (r**0.5), s * (r**0.5)
            base.append([-w / 2, -h / 2, w / 2, h / 2])
    base = torch.tensor(base, device=device)  # (k, 4)
    ys = (torch.arange(feat_h, device=device) + 0.5) * stride
    xs = (torch.arange(feat_w, device=device) + 0.5) * stride
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([cx, cy, cx, cy], dim=-1).view(-1, 1, 4)  # (H*W, 1, 4)
    return (centers + base[None]).view(-1, 4)


# ----------------------------------------------------------------------------
# RoI Pooling (Fast R-CNN과 동일 개념; adaptive max pooling으로 간결하게)
# ----------------------------------------------------------------------------
def roi_pool(features: torch.Tensor, rois: torch.Tensor, output_size: int, stride: int) -> torch.Tensor:
    """features (N,C,Hf,Wf), rois (R,5)=(batch_idx,x1,y1,x2,y2) -> (R,C,out,out). 좌표는 stride로 나눠 반올림."""
    _, _, Hf, Wf = features.shape
    outs = []
    for r in rois:
        b = int(r[0].item())
        x1, y1, x2, y2 = [int(round(v.item() / stride)) for v in r[1:]]
        x1, y1 = max(min(x1, Wf - 1), 0), max(min(y1, Hf - 1), 0)
        x2, y2 = max(min(x2, Wf - 1), x1), max(min(y2, Hf - 1), y1)
        patch = features[b : b + 1, :, y1 : y2 + 1, x1 : x2 + 1]
        outs.append(F.adaptive_max_pool2d(patch, output_size))
    return torch.cat(outs, dim=0)


# ----------------------------------------------------------------------------
# Region Proposal Network
# ----------------------------------------------------------------------------
class RPN(nn.Module):
    """3x3 conv(256) + ReLU -> 1x1 cls(2k) / 1x1 reg(4k). 모든 위치에서 가중치를 공유하는 fully-convolutional 구조."""

    def __init__(self, in_ch: int = 256, mid_ch: int = 256, num_anchors: int = 9):
        super().__init__()
        self.k = num_anchors
        self.conv = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.cls = nn.Conv2d(mid_ch, 2 * num_anchors, 1)
        self.reg = nn.Conv2d(mid_ch, 4 * num_anchors, 1)
        for m in (self.conv, self.cls, self.reg):
            nn.init.normal_(m.weight, std=0.01)  # 논문: 평균 0, 표준편차 0.01
            nn.init.zeros_(m.bias)

    def forward(self, fmap: torch.Tensor):
        N, _, H, W = fmap.shape
        h = F.relu(self.conv(fmap))
        logits = self.cls(h).permute(0, 2, 3, 1).reshape(N, H * W * self.k, 2)  # [bg, fg]
        deltas = self.reg(h).permute(0, 2, 3, 1).reshape(N, H * W * self.k, 4)
        return logits, deltas


def assign_anchor_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    img_h: int,
    img_w: int,
    pos_iou: float = 0.7,
    neg_iou: float = 0.3,
    batch_size: int = 256,
    pos_fraction: float = 0.5,
):
    """
    라벨: 1(양성) / 0(음성) / -1(무시).
      양성: (i) 어떤 GT와 IoU가 가장 큰 앵커, (ii) 어떤 GT와 IoU >= 0.7
      음성: 모든 GT와 IoU < 0.3 / 이미지 경계를 넘는 앵커는 학습에서 제외(무시)
    256개를 샘플링(양성 최대 128개, 모자라면 음성으로 채움). 반환: labels (A,), reg_targets (A,4)
    """
    A = anchors.shape[0]
    labels = torch.full((A,), -1, dtype=torch.long, device=anchors.device)
    inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= img_w) & (anchors[:, 3] <= img_h)

    iou = box_iou(anchors, gt_boxes)  # (A, G)
    iou[~inside] = -1.0
    max_iou, argmax = iou.max(dim=1)
    labels[(max_iou >= 0) & (max_iou < neg_iou)] = 0
    gt_best = iou.max(dim=0).values  # GT별 최대 IoU
    best_anchor = ((iou == gt_best[None, :]) & (gt_best[None, :] > 0)).any(dim=1)
    labels[best_anchor] = 1
    labels[max_iou >= pos_iou] = 1

    max_pos = int(batch_size * pos_fraction)
    pos = (labels == 1).nonzero().flatten()
    if len(pos) > max_pos:
        labels[pos[torch.randperm(len(pos))[: len(pos) - max_pos]]] = -1
    n_pos = int((labels == 1).sum())
    neg = (labels == 0).nonzero().flatten()
    if len(neg) > batch_size - n_pos:
        labels[neg[torch.randperm(len(neg))[: len(neg) - (batch_size - n_pos)]]] = -1

    targets = encode_boxes(anchors, gt_boxes[argmax])
    return labels, targets


def rpn_loss(logits, deltas, labels, targets):
    """한 이미지의 RPN 손실 = cross-entropy(샘플된 앵커) + smooth L1(양성 앵커만), 샘플 수로 정규화."""
    valid = labels >= 0
    cls_loss = F.cross_entropy(logits[valid], labels[valid])
    pos = labels == 1
    n = max(int(valid.sum()), 1)
    reg_loss = F.smooth_l1_loss(deltas[pos], targets[pos], beta=1.0 / 9, reduction="sum") / n
    return cls_loss, reg_loss


def generate_proposals(
    anchors, logits, deltas, img_h, img_w, pre_nms: int = 600, post_nms: int = 300, nms_thresh: float = 0.7, min_size: int = 8
) -> torch.Tensor:
    """한 이미지: fg 확률 상위 pre_nms개 -> 디코딩/클리핑 -> 작은 박스 제거 -> NMS(0.7) -> 상위 post_nms개."""
    scores = logits.softmax(dim=-1)[:, 1]
    top = scores.topk(min(pre_nms, scores.numel())).indices
    boxes = clip_boxes(decode_boxes(anchors[top], deltas[top]), img_h, img_w)
    scores = scores[top]
    ok = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    boxes, scores = boxes[ok], scores[ok]
    keep = nms(boxes, scores, nms_thresh)[:post_nms]
    return boxes[keep]


# ----------------------------------------------------------------------------
# 검출 헤드 (Fast R-CNN)
# ----------------------------------------------------------------------------
def sample_rois(proposals, gt_boxes, gt_labels, rois_per_image=128, pos_fraction=0.25, pos_iou=0.5):
    """제안 + GT 박스를 후보로 하여 양성(IoU>=0.5, 최대 25%)/배경(IoU<0.5)을 뽑고 라벨·회귀 타깃 생성."""
    cand = torch.cat([proposals, gt_boxes])
    max_iou, idx = box_iou(cand, gt_boxes).max(dim=1)
    pos = (max_iou >= pos_iou).nonzero().flatten()
    bg = (max_iou < pos_iou).nonzero().flatten()
    n_pos = min(int(rois_per_image * pos_fraction), len(pos))
    n_bg = min(rois_per_image - n_pos, len(bg))
    pos = pos[torch.randperm(len(pos))[:n_pos]]
    bg = bg[torch.randperm(len(bg))[:n_bg]]
    keep = torch.cat([pos, bg])
    labels = gt_labels[idx[keep]].clone()
    labels[n_pos:] = 0
    return cand[keep], labels, encode_boxes(cand[keep], gt_boxes[idx[keep]])


def detection_loss(cls_logits, box_deltas, labels, targets):
    cls_loss = F.cross_entropy(cls_logits, labels)
    pos = labels > 0
    if pos.any():
        pred = box_deltas[pos, labels[pos] - 1]
        loc_loss = F.smooth_l1_loss(pred, targets[pos] / targets.new_tensor(BOX_STD), reduction="sum") / labels.numel()
    else:
        loc_loss = box_deltas.sum() * 0.0
    return cls_loss, loc_loss


# ----------------------------------------------------------------------------
# 전체 모델
# ----------------------------------------------------------------------------
def make_backbone() -> nn.Sequential:
    """총 stride 16의 작은 conv 백본. 출력 채널 256."""

    def block(cin, cout, pool):
        layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return layers

    return nn.Sequential(*block(3, 32, True), *block(32, 64, True), *block(64, 128, True), *block(128, 256, True), *block(256, 256, False))


class FasterRCNN(nn.Module):
    """
    conv 특징 맵을 RPN과 검출 헤드가 공유한다.
      forward(images, gt_boxes=None, gt_labels=None):
        학습(train 모드 + GT 제공): 손실 dict 반환 (rpn_cls, rpn_reg, det_cls, det_reg)
        추론: 이미지별 (boxes, scores, labels) 리스트 반환
    labels는 1..num_classes (0은 배경).
    """

    def __init__(self, num_classes: int = 20, stride: int = 16, pool_size: int = 7, fc_dim: int = 1024):
        super().__init__()
        self.num_classes, self.stride, self.pool_size = num_classes, stride, pool_size
        self.backbone = make_backbone()
        self.rpn = RPN(256, 256, 9)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pool_size**2, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def head(self, fmap, rois):
        h = self.fc(roi_pool(fmap, rois, self.pool_size, self.stride))
        return self.cls_score(h), self.bbox_pred(h).view(-1, self.num_classes, 4)

    def forward(self, images, gt_boxes: List[torch.Tensor] = None, gt_labels: List[torch.Tensor] = None):
        N, _, H, W = images.shape
        fmap = self.backbone(images)
        anchors = generate_anchors(fmap.shape[2], fmap.shape[3], self.stride, device=images.device)
        rpn_logits, rpn_deltas = self.rpn(fmap)
        training = self.training and gt_boxes is not None

        pre, post = (600, 300) if training else (1000, 300)
        proposals = [
            generate_proposals(anchors, rpn_logits[i].detach(), rpn_deltas[i].detach(), H, W, pre, post) for i in range(N)
        ]  # detach: 근사 joint 학습

        if training:
            losses = dict(rpn_cls=0.0, rpn_reg=0.0)
            rois, labels, targets = [], [], []
            for i in range(N):
                lab, tgt = assign_anchor_targets(anchors, gt_boxes[i], H, W)
                c, r = rpn_loss(rpn_logits[i], rpn_deltas[i], lab, tgt)
                losses["rpn_cls"] = losses["rpn_cls"] + c / N
                losses["rpn_reg"] = losses["rpn_reg"] + r / N
                p, l, t = sample_rois(proposals[i], gt_boxes[i], gt_labels[i])
                rois.append(torch.cat([torch.full((len(p), 1), float(i), device=p.device), p], dim=1))
                labels.append(l)
                targets.append(t)
            cls_logits, box_deltas = self.head(fmap, torch.cat(rois))
            losses["det_cls"], losses["det_reg"] = detection_loss(
                cls_logits, box_deltas, torch.cat(labels), torch.cat(targets)
            )
            return losses

        results = []
        std = images.new_tensor(BOX_STD)
        for i in range(N):
            rois = torch.cat([torch.full((len(proposals[i]), 1), float(i), device=images.device), proposals[i]], dim=1)
            cls_logits, box_deltas = self.head(fmap, rois)
            probs = cls_logits.softmax(-1)
            out_b, out_s, out_l = [], [], []
            for c in range(1, self.num_classes + 1):  # 클래스별 NMS
                s = probs[:, c]
                m = s > 0.05
                if not m.any():
                    continue
                b = clip_boxes(decode_boxes(proposals[i][m], box_deltas[m, c - 1] * std), H, W)
                k = nms(b, s[m], 0.3)
                out_b.append(b[k])
                out_s.append(s[m][k])
                out_l.append(torch.full((len(k),), c))
            if out_b:
                results.append((torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)))
            else:
                results.append((torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
        return results


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FasterRCNN(num_classes=20)
    images = torch.randn(2, 3, 256, 320)

    fmap = model.backbone(images)
    print("feature map shape:", fmap.shape)  # [2, 256, 16, 20]
    anchors = generate_anchors(16, 20)
    print("anchors shape:", anchors.shape)  # [2880, 4]  (16*20*9)
    logits, deltas = model.rpn(fmap)
    print("rpn logits/deltas shape:", logits.shape, deltas.shape)  # [2, 2880, 2] [2, 2880, 4]

    gt_boxes = [torch.tensor([[40.0, 30.0, 160.0, 150.0]]), torch.tensor([[180.0, 90.0, 300.0, 230.0]])]
    gt_labels = [torch.tensor([5]), torch.tensor([12])]
    labels, _ = assign_anchor_targets(anchors, gt_boxes[0], 256, 320)
    print("anchor labels pos/neg/ignore:", (labels == 1).sum().item(), (labels == 0).sum().item(), (labels == -1).sum().item())

    model.train()
    losses = model(images, gt_boxes, gt_labels)
    print("train losses:", {k: round(v.item(), 4) for k, v in losses.items()})

    model.eval()
    with torch.no_grad():
        dets = model(images)
    print("eval outputs per image:", [tuple(d[0].shape) for d in dets])
