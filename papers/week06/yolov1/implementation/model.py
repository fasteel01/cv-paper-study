"""
YOLOv1 (Redmon et al., 2016) — from-scratch PyTorch implementation.

구성요소: 24 conv + 2 FC 네트워크(448x448 입력 -> S x S x (B*5 + C) = 7x7x30 출력),
GT 박스 -> 격자 타깃 인코딩, 논문의 sum-squared error 멀티파트 손실, 디코딩 + NMS 후처리.

원 논문과의 차이점(현대적 단순화):
  - ImageNet 사전학습(앞 20개 conv)을 하지 않고 처음부터 구성한다. 학습 가능한 구조만 동일하게 재현.
  - conv 층 뒤에 BatchNorm을 넣지 않는다(원 논문도 없음). 대신 초기화는 PyTorch 기본값을 쓴다.
  - 마지막 층은 논문처럼 선형(linear) 활성화이며, 나머지는 leaky ReLU(0.1). 출력 채널 구성은 아래와 같다:
        채널 0..4  : box1 (x, y, w, h, conf), 채널 5..9 : box2, 채널 10..29 : 클래스 확률(20)
  - 논문은 (x, y)를 "셀 내 상대 좌표", (w, h)를 "이미지 대비 비율"로 둔다. 동일하게 구현했다.
  - 박스 입력 형식은 정규화 (cx, cy, w, h) in [0, 1]. 한 셀에는 하나의 객체만 할당된다(논문의 제약).
"""

from typing import Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 네트워크
# ----------------------------------------------------------------------------
def _conv(cin, cout, k, s=1):
    return [nn.Conv2d(cin, cout, k, s, padding=k // 2), nn.LeakyReLU(0.1, inplace=True)]


def make_conv_layers() -> nn.Sequential:
    """논문 Figure 3의 24개 conv. (kernel, out_channels) 표기, 'M'은 2x2 stride 2 maxpool, 리스트는 반복."""
    layers = []
    layers += _conv(3, 64, 7, 2) + [nn.MaxPool2d(2, 2)]  # 448 -> 224 -> 112
    layers += _conv(64, 192, 3) + [nn.MaxPool2d(2, 2)]  # 56
    layers += _conv(192, 128, 1) + _conv(128, 256, 3) + _conv(256, 256, 1) + _conv(256, 512, 3) + [nn.MaxPool2d(2, 2)]  # 28
    for _ in range(4):
        layers += _conv(512, 256, 1) + _conv(256, 512, 3)
    layers += _conv(512, 512, 1) + _conv(512, 1024, 3) + [nn.MaxPool2d(2, 2)]  # 14
    for _ in range(2):
        layers += _conv(1024, 512, 1) + _conv(512, 1024, 3)
    layers += _conv(1024, 1024, 3) + _conv(1024, 1024, 3, 2)  # 7
    layers += _conv(1024, 1024, 3) + _conv(1024, 1024, 3)
    return nn.Sequential(*layers)


class YOLOv1(nn.Module):
    """forward(x): (N, 3, 448, 448) -> (N, S, S, B*5 + C)."""

    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.features = make_conv_layers()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),  # 선형 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x))
        return out.view(-1, self.S, self.S, self.B * 5 + self.C)


# ----------------------------------------------------------------------------
# 타깃 인코딩 / IoU
# ----------------------------------------------------------------------------
def encode_target(boxes: torch.Tensor, labels: torch.Tensor, S: int = 7, B: int = 2, C: int = 20) -> torch.Tensor:
    """
    boxes: (G, 4) 정규화 (cx, cy, w, h), labels: (G,) in [0, C).
    반환: (S, S, 5 + C) = (x, y, w, h, obj, one-hot(C)); x, y는 셀 내 상대 좌표, 객체 없는 셀은 obj=0.
    (예측 B개 중 어느 것이 "담당"인지는 손실 계산 시 IoU로 결정하므로 타깃은 박스 하나만 가진다.)
    """
    target = torch.zeros(S, S, 5 + C)
    for (cx, cy, w, h), lab in zip(boxes.tolist(), labels.tolist()):
        j, i = min(int(cx * S), S - 1), min(int(cy * S), S - 1)  # 열, 행
        if target[i, j, 4] == 1:  # 셀당 객체 1개 제약: 먼저 온 객체 유지
            continue
        target[i, j, :5] = torch.tensor([cx * S - j, cy * S - i, w, h, 1.0])
        target[i, j, 5 + lab] = 1.0
    return target


def cell_boxes_to_xyxy(box: torch.Tensor, S: int) -> torch.Tensor:
    """box: (N, S, S, 4)=(x_cell, y_cell, w, h) -> 이미지 정규화 (x1, y1, x2, y2)."""
    N = box.shape[0]
    gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    cx = (box[..., 0] + gx.to(box)) / S
    cy = (box[..., 1] + gy.to(box)) / S
    w, h = box[..., 2], box[..., 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1).view(N, S, S, 4)


def iou_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (..., 4) xyxy 의 원소별 IoU."""
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


# ----------------------------------------------------------------------------
# 손실
# ----------------------------------------------------------------------------
def yolo_loss(pred: torch.Tensor, target: torch.Tensor, S=7, B=2, C=20, l_coord=5.0, l_noobj=0.5) -> Tuple[torch.Tensor, dict]:
    """
    논문의 식 (3): 모두 sum-squared error.
      l_coord * sum 1_obj [(x-x^)^2 + (y-y^)^2]                       # 담당 predictor의 중심
    + l_coord * sum 1_obj [(sqrt w - sqrt w^)^2 + (sqrt h - sqrt h^)^2]  # 크기(큰 박스의 오차를 덜 민감하게)
    + sum 1_obj (C - C^)^2                                             # 객체 있는 predictor의 confidence (타깃 = IoU)
    + l_noobj * sum 1_noobj (C - C^)^2                                 # 객체 없는 predictor (타깃 = 0)
    + sum 1_obj_cell sum_c (p(c) - p^(c))^2                            # 클래스 확률(객체가 있는 셀만)
    "담당(responsible)" predictor = 해당 셀의 B개 예측 중 GT와 IoU가 가장 큰 것.
    pred: (N,S,S,B*5+C), target: (N,S,S,5+C). 반환: (배치 평균 손실, 항목별 dict)
    """
    N = pred.shape[0]
    obj_cell = target[..., 4] == 1  # (N,S,S)
    gt_xyxy = cell_boxes_to_xyxy(target[..., :4], S)  # (N,S,S,4)

    pred_boxes = pred[..., : B * 5].view(N, S, S, B, 5)
    ious = torch.stack(
        [iou_elementwise(cell_boxes_to_xyxy(pred_boxes[..., b, :4], S), gt_xyxy) for b in range(B)], dim=-1
    )  # (N,S,S,B)
    best = ious.argmax(dim=-1)  # 담당 predictor 인덱스 (N,S,S)
    resp = torch.zeros_like(ious, dtype=torch.bool)
    resp.scatter_(-1, best.unsqueeze(-1), True)
    resp = resp & obj_cell.unsqueeze(-1)  # 1_obj_ij  (N,S,S,B)
    noobj = ~resp  # 1_noobj_ij: 담당이 아닌 모든 predictor

    # 좌표 손실
    px, py, pw, ph, pconf = pred_boxes.unbind(-1)
    t = target[..., :4].unsqueeze(-2)  # (N,S,S,1,4)
    xy_loss = (((px - t[..., 0]) ** 2 + (py - t[..., 1]) ** 2) * resp).sum()
    # sqrt 는 예측이 음수일 수 있어 부호를 유지한 안정화 형태(sign * sqrt(|.| + eps))를 쓴다.
    sq = lambda v: torch.sign(v) * torch.sqrt(v.abs() + 1e-6)
    wh_loss = (((sq(pw) - torch.sqrt(t[..., 2])) ** 2 + (sq(ph) - torch.sqrt(t[..., 3])) ** 2) * resp).sum()

    # confidence: 담당 predictor 타깃은 (예측박스, GT) IoU(그래디언트 차단), 나머지는 0
    conf_obj = (((pconf - ious.detach()) ** 2) * resp).sum()
    conf_noobj = ((pconf**2) * noobj).sum()

    # 클래스 확률: 객체가 있는 셀에서만
    cls_loss = (((pred[..., B * 5 :] - target[..., 5:]) ** 2).sum(-1) * obj_cell).sum()

    parts = dict(xy=l_coord * xy_loss, wh=l_coord * wh_loss, conf_obj=conf_obj, conf_noobj=l_noobj * conf_noobj, cls=cls_loss)
    total = sum(parts.values()) / N
    return total, {k: (v / N).item() for k, v in parts.items()}


# ----------------------------------------------------------------------------
# 후처리: 디코딩 + NMS
# ----------------------------------------------------------------------------
@torch.no_grad()
def decode_predictions(pred: torch.Tensor, S=7, B=2, C=20, conf_thresh=0.2, nms_iou=0.5):
    """
    한 배치의 예측 -> 이미지별 (boxes_xyxy[정규화], scores, labels) 리스트.
    클래스별 점수 = Pr(class|obj) * confidence (논문 식 (1)). 클래스별 NMS 적용.
    """
    N = pred.shape[0]
    boxes_all, scores_all = [], []
    for b in range(B):
        boxes_all.append(cell_boxes_to_xyxy(pred[..., b * 5 : b * 5 + 4], S).view(N, -1, 4))
        conf = pred[..., b * 5 + 4].reshape(N, -1, 1)
        cls_p = pred[..., B * 5 :].reshape(N, -1, C)
        scores_all.append(conf * cls_p)  # (N, S*S, C)
    boxes = torch.cat(boxes_all, dim=1)  # (N, B*S*S, 4)
    scores = torch.cat(scores_all, dim=1)  # (N, B*S*S, C)

    results = []
    for n in range(N):
        out_b, out_s, out_l = [], [], []
        for c in range(C):
            m = scores[n, :, c] > conf_thresh
            if not m.any():
                continue
            bx, sc = boxes[n][m], scores[n, :, c][m]
            order = sc.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0]
                keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]
                iou = iou_elementwise(bx[i].expand(len(rest), 4), bx[rest])
                order = rest[iou <= nms_iou]
            keep = torch.stack(keep)
            out_b.append(bx[keep])
            out_s.append(sc[keep])
            out_l.append(torch.full((len(keep),), c))
        if out_b:
            results.append((torch.cat(out_b), torch.cat(out_s), torch.cat(out_l)))
        else:
            results.append((torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    model = YOLOv1()
    n_conv = sum(isinstance(m, nn.Conv2d) for m in model.features)
    n_fc = sum(isinstance(m, nn.Linear) for m in model.head)
    print(f"conv layers: {n_conv}, fc layers: {n_fc}")  # 24, 2
    print("params:", sum(p.numel() for p in model.parameters()))

    x = torch.randn(2, 3, 448, 448)
    print("feature map shape:", model.features(x).shape)  # [2, 1024, 7, 7]
    out = model(x)
    print("output shape:", out.shape)  # [2, 7, 7, 30]

    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.2, 0.3, 0.2, 0.2]])
    labels = torch.tensor([14, 6])
    target = encode_target(boxes, labels)
    print("target shape:", target.shape, "objects:", int(target[..., 4].sum()))  # [7,7,25], 2
    loss, parts = yolo_loss(out, torch.stack([target, target]))
    print("loss:", round(loss.item(), 4), {k: round(v, 3) for k, v in parts.items()})

    dets = decode_predictions(out.detach(), conf_thresh=0.0)
    print("detections image0:", dets[0][0].shape)
