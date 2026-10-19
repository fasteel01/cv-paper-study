"""
R-CNN (Girshick et al., 2014) — from-scratch PyTorch implementation of the core pieces.

구성요소: IoU, 박스 인코딩/디코딩, 영역 warp, NMS, AlexNet 계열 특징 추출기, 분류/회귀 헤드.

원 논문과의 차이점(현대적 단순화):
  - Selective Search는 구현하지 않는다. 제안 박스는 외부에서 입력으로 받는다(더미 박스로 시연).
  - 클래스별 선형 SVM + hard negative mining 대신, 미세조정 CNN 위의 선형 분류층(softmax, N+1 way)을 사용한다.
  - bbox regression은 pool5 특징에 대한 릿지 회귀 대신 fc7 위의 선형층(smooth L1로 학습)으로 구성한다.
  - 원 논문의 2-GPU 분할 AlexNet 대신 단일 경로 AlexNet 구조(LRN 생략)를 쓴다.
  - 박스 좌표는 (x1, y1, x2, y2) 픽셀 단위.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 박스 유틸리티
# ----------------------------------------------------------------------------
def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, 4), b: (M, 4) [x1,y1,x2,y2] -> IoU 행렬 (N, M)"""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def encode_boxes(proposals: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """제안 박스 P -> 정답 박스 G 로의 회귀 타깃 (tx, ty, tw, th). 논문 식 (6)~(9)."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gx = gt[:, 0] + 0.5 * gw
    gy = gt[:, 1] + 0.5 * gh
    return torch.stack([(gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)], dim=1)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """encode_boxes의 역변환: 제안 박스 + 예측 delta -> 보정된 박스."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gx = pw * deltas[:, 0] + px
    gy = ph * deltas[:, 1] + py
    gw = pw * torch.exp(deltas[:, 2])
    gh = ph * torch.exp(deltas[:, 3])
    return torch.stack([gx - 0.5 * gw, gy - 0.5 * gh, gx + 0.5 * gw, gy + 0.5 * gh], dim=1)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.3) -> torch.Tensor:
    """Greedy NMS. 유지할 박스의 인덱스를 점수 내림차순으로 반환."""
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)


# ----------------------------------------------------------------------------
# 영역 warp
# ----------------------------------------------------------------------------
def warp_region(image: torch.Tensor, box: torch.Tensor, out_size: int = 227, context: int = 16) -> torch.Tensor:
    """
    image: (3, H, W), box: (4,) -> (3, out_size, out_size)

    논문 방식: 타이트 박스를 out_size로 비등방 리사이즈할 때, 출력 기준으로 `context` 픽셀만큼의
    주변 문맥이 포함되도록 원본 박스를 확장한다. 이미지 밖은 0(평균 제거된 입력 기준 평균값)으로 채운다.
    """
    _, H, W = image.shape
    x1, y1, x2, y2 = box.tolist()
    # 출력에서 context 픽셀 = 원본에서 context * (박스크기 / (out_size - 2*context)) 픽셀
    inner = out_size - 2 * context
    pad_x = context * (x2 - x1) / inner
    pad_y = context * (y2 - y1) / inner
    x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    # 이미지 안쪽 부분만 crop 하고, 나머지는 0 패딩 (좌표계를 유지하려고 확장된 캔버스에 붙여넣는다)
    cx1, cy1 = max(int(round(x1)), 0), max(int(round(y1)), 0)
    cx2, cy2 = min(int(round(x2)), W), min(int(round(y2)), H)
    ex1, ey1, ex2, ey2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    ew, eh = max(ex2 - ex1, 1), max(ey2 - ey1, 1)
    canvas = image.new_zeros(3, eh, ew)
    if cx2 > cx1 and cy2 > cy1:
        canvas[:, cy1 - ey1 : cy2 - ey1, cx1 - ex1 : cx2 - ex1] = image[:, cy1:cy2, cx1:cx2]
    out = F.interpolate(canvas[None], size=(out_size, out_size), mode="bilinear", align_corners=False)
    return out[0]


# ----------------------------------------------------------------------------
# 네트워크
# ----------------------------------------------------------------------------
class RCNNBackbone(nn.Module):
    """AlexNet 계열 CNN: conv1~5 + fc6, fc7. 입력 (N,3,227,227) -> 4096-d 특징."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),  # 227 -> 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 27
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 13
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # pool5: 256 x 6 x 6
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class RCNN(nn.Module):
    """
    영역(이미 warp된 227x227 패치)들을 입력받아 (분류 logits, 박스 delta)를 출력한다.
      cls_logits: (R, num_classes + 1)  (index 0 = 배경)
      box_deltas: (R, num_classes, 4)   클래스별 회귀
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = RCNNBackbone()
        self.cls_head = nn.Linear(4096, num_classes + 1)
        self.box_head = nn.Linear(4096, num_classes * 4)

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(patches)
        return self.cls_head(feat), self.box_head(feat).view(-1, self.num_classes, 4)

    @torch.no_grad()
    def detect(
        self,
        image: torch.Tensor,
        proposals: torch.Tensor,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        chunk: int = 64,
    ):
        """
        추론 파이프라인: 각 제안을 warp -> CNN -> 클래스별 점수/박스 보정 -> 클래스별 NMS.
        (제안마다 CNN을 독립 실행하므로 느리다 — Fast R-CNN이 해결하려는 점)
        반환: list of (box[4], score, class_id(1-based))
        """
        self.eval()
        patches = torch.stack([warp_region(image, b) for b in proposals])
        logits, deltas = [], []
        for i in range(0, len(patches), chunk):
            l, d = self(patches[i : i + chunk])
            logits.append(l)
            deltas.append(d)
        probs = F.softmax(torch.cat(logits), dim=1)
        deltas = torch.cat(deltas)

        results = []
        for c in range(1, self.num_classes + 1):
            scores = probs[:, c]
            m = scores > score_thresh
            if m.sum() == 0:
                continue
            boxes = decode_boxes(proposals[m], deltas[m, c - 1])
            keep = nms(boxes, scores[m], nms_thresh)
            for k in keep:
                results.append((boxes[k], scores[m][k].item(), c))
        return results


def rcnn_loss(cls_logits, box_deltas, labels, reg_targets, lambda_reg: float = 1.0):
    """
    labels: (R,) 0=배경, 1..N=클래스. reg_targets: (R,4) (양성에만 의미 있음).
    분류: cross-entropy, 회귀: 양성 샘플의 해당 클래스 delta에 대한 smooth L1.
    """
    cls_loss = F.cross_entropy(cls_logits, labels)
    pos = labels > 0
    if pos.any():
        pred = box_deltas[pos, labels[pos] - 1]
        reg_loss = F.smooth_l1_loss(pred, reg_targets[pos], reduction="sum") / max(int(pos.sum()), 1)
    else:
        reg_loss = box_deltas.sum() * 0.0
    return cls_loss + lambda_reg * reg_loss, cls_loss.detach(), reg_loss.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RCNN(num_classes=20)

    # 1) 패치 단위 forward
    patches = torch.randn(4, 3, 227, 227)
    logits, deltas = model(patches)
    print("cls_logits shape:", logits.shape)  # [4, 21]
    print("box_deltas shape:", deltas.shape)  # [4, 20, 4]

    # 2) 박스 유틸리티 점검
    p = torch.tensor([[10.0, 10.0, 110.0, 90.0]])
    g = torch.tensor([[20.0, 15.0, 120.0, 100.0]])
    t = encode_boxes(p, g)
    print("IoU:", box_iou(p, g).item())
    print("decode(encode) == gt:", torch.allclose(decode_boxes(p, t), g, atol=1e-4))

    # 3) warp + 전체 추론 파이프라인 (더미 이미지/제안)
    image = torch.randn(3, 300, 400)
    proposals = torch.tensor([[50.0, 40.0, 200.0, 180.0], [-10.0, -10.0, 90.0, 80.0], [200.0, 100.0, 390.0, 290.0]])
    print("warped patch shape:", warp_region(image, proposals[1]).shape)  # [3, 227, 227]
    dets = model.detect(image, proposals, score_thresh=0.0)
    print("num detections:", len(dets))
