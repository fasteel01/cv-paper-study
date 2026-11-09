"""
RetinaNet 학습 step 데모 (더미 데이터).

COCO / VOC 는 무겁기 때문에 더미 이미지와 GT 박스로
  1) ResNet-50 + FPN(P3~P7) + 서브넷으로 모든 앵커의 클래스 로짓/박스 델타 계산
  2) 앵커 매칭: IoU >= 0.5 positive, < 0.4 negative, 그 사이는 무시(ignore)
  3) Focal Loss(alpha=0.25, gamma=2)를 "모든" 유효 앵커에 합산하고 positive 앵커 수로 정규화
     + positive 앵커에 대한 smooth L1 박스 회귀 손실
  4) 역전파
를 보여준다. 마지막에 같은 로짓에 대해 Focal Loss 와 일반 BCE 의 손실 구성(easy negative 가 차지하는 비중)을 비교한다.

단순화: 박스 회귀는 논문의 smooth L1 (beta=1/9), 이미지 경계 밖 앵커도 그대로 사용.

실행:
    python train.py
"""

import torch
import torch.nn.functional as F

from model import RetinaNet, box_iou, encode_boxes, sigmoid_focal_loss

NUM_CLASSES = 20


def make_dummy_batch(n=2, h=256, w=384):
    images = torch.randn(n, 3, h, w)
    boxes = [torch.tensor([[40.0, 30.0, 160.0, 150.0], [200.0, 90.0, 340.0, 230.0]]) for _ in range(n)]
    labels = [torch.tensor([5, 12]) for _ in range(n)]
    return images, boxes, labels


def match_anchors(anchors, gt_boxes, gt_labels, pos_thr=0.5, neg_thr=0.4):
    """앵커별 상태. labels: -2 = ignore, -1 = background, >=0 = 클래스. 그리고 매칭된 GT 박스."""
    iou = box_iou(gt_boxes, anchors)  # (G, A)
    max_iou, argmax = iou.max(dim=0)
    labels = torch.full((anchors.size(0),), -2, dtype=torch.long, device=anchors.device)
    labels[max_iou < neg_thr] = -1
    pos = max_iou >= pos_thr
    labels[pos] = gt_labels[argmax[pos]]
    return labels, gt_boxes[argmax]


def retinanet_loss(cls_logits, box_deltas, anchors, gt_boxes_list, gt_labels_list):
    """반환: (cls_loss, box_loss, 정규화 상수 = positive 앵커 수)"""
    cls_loss, box_loss, num_pos_total = 0.0, 0.0, 0
    for i, (gt, gl) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        labels, matched = match_anchors(anchors, gt, gl)
        valid = labels != -2
        pos = labels >= 0
        target = torch.zeros_like(cls_logits[i])
        target[pos, labels[pos]] = 1.0  # one-hot, negative 는 전부 0
        # 모든 유효 앵커(수만 개) 에 대해 합산 -> Focal Loss 가 easy negative 를 억제하지 않으면 이 합을 negative 가 지배
        cls_loss = cls_loss + sigmoid_focal_loss(cls_logits[i][valid], target[valid], reduction="sum")
        if pos.any():
            tgt = encode_boxes(matched[pos], anchors[pos])
            box_loss = box_loss + F.smooth_l1_loss(box_deltas[i][pos], tgt, beta=1 / 9, reduction="sum")
        num_pos_total += int(pos.sum())
    norm = max(num_pos_total, 1)
    return cls_loss / norm, box_loss / norm, norm


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaNet(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    model.train()
    for step in range(1, 4):
        images, gt_boxes, gt_labels = make_dummy_batch()
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        cls, box, anchors, _ = model(images)
        l_cls, l_box, n_pos = retinanet_loss(cls, box, anchors, gt_boxes, gt_labels)
        total = l_cls + l_box
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        print(f"[step {step}] total={total.item():.4f} focal_cls={l_cls.item():.4f} box={l_box.item():.4f} "
              f"(positive anchors: {n_pos}, total anchors/img: {anchors.size(0)})")

    # Focal Loss vs BCE: easy negative 가 손실에서 차지하는 비중
    with torch.no_grad():
        labels, _ = match_anchors(anchors, gt_boxes[0], gt_labels[0])
        valid, pos = labels != -2, labels >= 0
        target = torch.zeros_like(cls[0])
        target[pos, labels[pos]] = 1.0
        bce = F.binary_cross_entropy_with_logits(cls[0][valid], target[valid], reduction="none")
        fl = sigmoid_focal_loss(cls[0][valid], target[valid])
        neg_mask = (target[valid] == 0)
        print(f"negative 항이 전체 손실에서 차지하는 비율: BCE {bce[neg_mask].sum() / bce.sum():.3f} "
              f"| Focal {fl[neg_mask].sum() / fl.sum():.3f}")


if __name__ == "__main__":
    main()
