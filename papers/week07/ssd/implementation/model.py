"""
SSD300 (Liu et al., 2016) — from-scratch PyTorch implementation.

구성: VGG-16 base(dilated fc6) + L2Norm(conv4_3) + extras + 6개 멀티스케일 head
      + default box 생성 / 매칭 / MultiBox 손실(hard negative mining) / 디코딩 + 클래스별 NMS

원 논문과의 차이(단순화):
  - ImageNet 사전학습 가중치 로딩 및 fc6/fc7 subsampling 초기화는 생략 (무작위 초기화)
  - VGG에 BatchNorm은 쓰지 않음(원 논문과 동일). 데이터 증강은 포함하지 않음
  - 박스 인코딩에 공식 구현의 variance (0.1, 0.2)를 사용
  - default box 크기는 공식 Caffe 구현의 값(30,60,111,162,213,264,315 / 300)을 사용
  - NMS / IoU 는 순수 PyTorch 로 직접 구현 (속도보다 가독성 우선)

박스 좌표 규약: 정규화된 (xmin, ymin, xmax, ymax) in [0,1] (corner), default box 는 (cx, cy, w, h).
클래스 규약: 0 = 배경, 1..C-1 = 객체 클래스.
"""

import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANCES = (0.1, 0.2)


# ---------------------------------------------------------------- box utils
def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], dim=-1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([(b[..., :2] + b[..., 2:]) / 2, b[..., 2:] - b[..., :2]], dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (A,4), b: (B,4) corner 형식 -> (A,B) IoU"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode(gt_xyxy: torch.Tensor, priors_cxcy: torch.Tensor) -> torch.Tensor:
    """GT(corner) -> default box 기준 회귀 타깃 (g_cxcy, g_wh)."""
    g = xyxy_to_cxcywh(gt_xyxy)
    d_cxcy = (g[:, :2] - priors_cxcy[:, :2]) / (VARIANCES[0] * priors_cxcy[:, 2:])
    d_wh = torch.log(g[:, 2:].clamp(min=1e-8) / priors_cxcy[:, 2:]) / VARIANCES[1]
    return torch.cat([d_cxcy, d_wh], dim=1)


def decode(loc: torch.Tensor, priors_cxcy: torch.Tensor) -> torch.Tensor:
    """예측 오프셋 -> corner 박스."""
    cxcy = priors_cxcy[:, :2] + loc[:, :2] * VARIANCES[0] * priors_cxcy[:, 2:]
    wh = priors_cxcy[:, 2:] * torch.exp(loc[:, 2:] * VARIANCES[1])
    return cxcywh_to_xyxy(torch.cat([cxcy, wh], dim=1))


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Greedy NMS. 유지할 인덱스 반환."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


# ---------------------------------------------------------------- default boxes
# (feature map size, box size min, box size max, extra aspect ratios) — image size 300 기준
FEATURE_SIZES = [38, 19, 10, 5, 3, 1]
MIN_SIZES = [30, 60, 111, 162, 213, 264]
MAX_SIZES = [60, 111, 162, 213, 264, 315]
ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
IMAGE_SIZE = 300


def boxes_per_location():
    return [2 + 2 * len(ar) for ar in ASPECT_RATIOS]  # [4, 6, 6, 6, 4, 4]


def generate_default_boxes(clip: bool = True) -> torch.Tensor:
    """(8732, 4) default box, (cx, cy, w, h) 정규화 좌표.
    순서: feature map -> row(i) -> col(j) -> box 종류. head 출력의 flatten 순서와 일치한다."""
    boxes = []
    for f, mn, mx, ars in zip(FEATURE_SIZES, MIN_SIZES, MAX_SIZES, ASPECT_RATIOS):
        for i, j in product(range(f), repeat=2):
            cx, cy = (j + 0.5) / f, (i + 0.5) / f
            s = mn / IMAGE_SIZE
            s_prime = math.sqrt(mn * mx) / IMAGE_SIZE
            boxes.append([cx, cy, s, s])  # a_r = 1
            boxes.append([cx, cy, s_prime, s_prime])  # a_r = 1, 스케일 s'_k
            for ar in ars:  # a_r, 1/a_r
                r = math.sqrt(ar)
                boxes.append([cx, cy, s * r, s / r])
                boxes.append([cx, cy, s / r, s * r])
    out = torch.tensor(boxes, dtype=torch.float32)
    if clip:
        out = out.clamp(0, 1)
    return out


# ---------------------------------------------------------------- network
class L2Norm(nn.Module):
    """채널 방향 L2 정규화 + 학습 가능한 채널별 스케일 (ParseNet). conv4_3 용."""

    def __init__(self, channels: int, scale: float = 20.0, eps: float = 1e-10):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), scale))
        self.eps = eps

    def forward(self, x):
        x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)
        return x * self.weight.view(1, -1, 1, 1)


def _vgg_base() -> nn.ModuleList:
    """VGG-16 conv1_1 ~ fc7. pool3 은 ceil_mode 로 75 -> 38 을 만든다."""
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512]
    layers, in_c = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        elif v == "C":
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(in_c, v, 3, padding=1), nn.ReLU(inplace=True)]
            in_c = v
    layers += [
        nn.MaxPool2d(3, 1, padding=1),  # pool5: 3x3 / stride 1
        nn.Conv2d(512, 1024, 3, padding=6, dilation=6),  # fc6 -> dilated conv
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 1),  # fc7 -> 1x1 conv
        nn.ReLU(inplace=True),
    ]
    return nn.ModuleList(layers)


class SSD300(nn.Module):
    """입력 (N,3,300,300) -> (loc (N,8732,4), conf (N,8732,num_classes))"""

    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.base = _vgg_base()
        # conv4_3 의 ReLU 직후 = base[22] (0-indexed, 아래 forward 참고)
        self.l2norm = L2Norm(512, 20.0)

        def extra(cin, mid, cout, stride, pad):
            return nn.Sequential(
                nn.Conv2d(cin, mid, 1), nn.ReLU(inplace=True),
                nn.Conv2d(mid, cout, 3, stride=stride, padding=pad), nn.ReLU(inplace=True),
            )

        self.extras = nn.ModuleList([
            extra(1024, 256, 512, 2, 1),  # conv8_2: 19 -> 10
            extra(512, 128, 256, 2, 1),  # conv9_2: 10 -> 5
            extra(256, 128, 256, 1, 0),  # conv10_2: 5 -> 3
            extra(256, 128, 256, 1, 0),  # conv11_2: 3 -> 1
        ])

        src_channels = [512, 1024, 512, 256, 256, 256]
        n_box = boxes_per_location()
        self.loc_heads = nn.ModuleList([nn.Conv2d(c, n * 4, 3, padding=1) for c, n in zip(src_channels, n_box)])
        self.conf_heads = nn.ModuleList([nn.Conv2d(c, n * num_classes, 3, padding=1) for c, n in zip(src_channels, n_box)])
        self.register_buffer("default_boxes", generate_default_boxes(), persistent=False)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.extras.modules()) + list(self.loc_heads.modules()) + list(self.conf_heads.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        sources = []
        for k, layer in enumerate(self.base):
            x = layer(x)
            if k == 22:  # conv4_3 + ReLU (pool4 직전)
                sources.append(self.l2norm(x))
        sources.append(x)  # fc7
        for ex in self.extras:
            x = ex(x)
            sources.append(x)

        n = x.size(0)
        locs, confs = [], []
        for s, lh, ch in zip(sources, self.loc_heads, self.conf_heads):
            # (N, A*k, H, W) -> (N, H, W, A*k) -> (N, H*W*A, k) : default box 순서와 일치
            locs.append(lh(s).permute(0, 2, 3, 1).reshape(n, -1, 4))
            confs.append(ch(s).permute(0, 2, 3, 1).reshape(n, -1, self.num_classes))
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    @torch.no_grad()
    def detect(self, images, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
        """이미지별 [(boxes (K,4), scores (K,), labels (K,))] 반환. 좌표는 정규화 corner."""
        self.eval()
        loc, conf = self(images)
        probs = F.softmax(conf, dim=-1)
        results = []
        for i in range(images.size(0)):
            boxes_all = decode(loc[i], self.default_boxes).clamp(0, 1)
            out_b, out_s, out_l = [], [], []
            for c in range(1, self.num_classes):
                sc = probs[i, :, c]
                m = sc > conf_thresh
                if not m.any():
                    continue
                b, s = boxes_all[m], sc[m]
                keep = nms(b, s, nms_thresh)
                out_b.append(b[keep]); out_s.append(s[keep])
                out_l.append(torch.full((len(keep),), c, dtype=torch.long))
            if out_b:
                b, s, l = torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)
                top = s.argsort(descending=True)[:top_k]
                results.append((b[top], s[top], l[top]))
            else:
                results.append((torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
        return results


# ---------------------------------------------------------------- matching + loss
def match(gt_boxes, gt_labels, default_boxes, iou_thr=0.5):
    """한 이미지에 대한 매칭.
    gt_boxes: (G,4) corner, gt_labels: (G,) in 1..C-1
    반환: loc_target (P,4), conf_target (P,) (0 = 배경)"""
    priors_xyxy = cxcywh_to_xyxy(default_boxes)
    iou = box_iou(gt_boxes, priors_xyxy)  # (G, P)
    best_prior_iou, best_prior_idx = iou.max(dim=1)  # 각 GT 에 가장 잘 맞는 default box
    best_gt_iou, best_gt_idx = iou.max(dim=0)  # 각 default box 에 가장 잘 맞는 GT
    # (1) 모든 GT 가 최소 1개 default box 를 갖도록 강제 매칭
    best_gt_iou[best_prior_idx] = 2.0
    for g in range(gt_boxes.size(0)):
        best_gt_idx[best_prior_idx[g]] = g
    # (2) IoU > thr 인 default box 도 매칭
    conf_target = gt_labels[best_gt_idx].clone()
    conf_target[best_gt_iou < iou_thr] = 0
    loc_target = encode(gt_boxes[best_gt_idx], default_boxes)
    return loc_target, conf_target


class MultiBoxLoss(nn.Module):
    """L = (L_conf + alpha * L_loc) / N, hard negative mining (neg:pos = 3:1)."""

    def __init__(self, iou_thr=0.5, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.iou_thr, self.neg_pos_ratio, self.alpha = iou_thr, neg_pos_ratio, alpha

    def forward(self, loc_pred, conf_pred, gt_boxes_list, gt_labels_list, default_boxes):
        n, p, c = conf_pred.shape
        loc_t = torch.zeros(n, p, 4, device=loc_pred.device)
        conf_t = torch.zeros(n, p, dtype=torch.long, device=loc_pred.device)
        for i in range(n):
            loc_t[i], conf_t[i] = match(gt_boxes_list[i], gt_labels_list[i], default_boxes, self.iou_thr)

        pos = conf_t > 0  # (N,P)
        num_pos = pos.sum(dim=1, keepdim=True)  # (N,1)
        loss_l = F.smooth_l1_loss(loc_pred[pos], loc_t[pos], reduction="sum")

        # hard negative mining: 배경 CE 가 큰 순으로 neg 선택
        ce = F.cross_entropy(conf_pred.reshape(-1, c), conf_t.reshape(-1), reduction="none").view(n, p)
        ce_neg = ce.clone()
        ce_neg[pos] = 0
        rank = ce_neg.argsort(dim=1, descending=True).argsort(dim=1)
        num_neg = (self.neg_pos_ratio * num_pos).clamp(max=p - 1)
        neg = rank < num_neg
        loss_c = ce[pos | neg].sum()

        N = num_pos.sum().clamp(min=1).float()
        return loss_c / N, self.alpha * loss_l / N


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SSD300(num_classes=21)
    x = torch.randn(2, 3, 300, 300)
    loc, conf = model(x)
    print("default boxes:", tuple(model.default_boxes.shape))  # (8732, 4)
    print("loc  :", tuple(loc.shape))  # (2, 8732, 4)
    print("conf :", tuple(conf.shape))  # (2, 8732, 21)
    assert loc.shape == (2, 8732, 4) and conf.shape == (2, 8732, 21)
    print("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    gts = [torch.tensor([[0.1, 0.1, 0.5, 0.6]]), torch.tensor([[0.3, 0.3, 0.9, 0.8], [0.05, 0.5, 0.3, 0.95]])]
    lbs = [torch.tensor([3]), torch.tensor([7, 12])]
    lc, ll = MultiBoxLoss()(loc, conf, gts, lbs, model.default_boxes)
    print(f"loss conf={lc.item():.4f} loc={ll.item():.4f}")
    dets = model.detect(x[:1], conf_thresh=0.05)
    print("detections (img0):", tuple(dets[0][0].shape))
