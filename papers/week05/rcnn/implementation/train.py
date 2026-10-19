"""
R-CNN 미세조정 단계 데모 (더미 데이터).

PASCAL VOC는 무겁고 Selective Search가 필요하므로, 여기서는 더미 이미지/GT/제안 박스로
  1) 제안 박스 라벨링 (IoU >= 0.5 -> GT 클래스, 그 외 -> 배경)
  2) 미니배치 샘플링 (양성 32 + 배경 96 = 128)
  3) 분류 + 박스 회귀 손실 계산, 1 step 학습
을 보여준다.

실행:
    python train.py
"""

import torch

from model import RCNN, box_iou, encode_boxes, rcnn_loss, warp_region

NUM_CLASSES = 20


def label_proposals(proposals, gt_boxes, gt_labels, pos_iou=0.5):
    """각 제안에 대해 가장 IoU가 큰 GT를 찾아 라벨(0=배경)과 회귀 타깃을 만든다."""
    iou = box_iou(proposals, gt_boxes)  # (P, G)
    max_iou, idx = iou.max(dim=1)
    labels = gt_labels[idx].clone()
    labels[max_iou < pos_iou] = 0
    targets = encode_boxes(proposals, gt_boxes[idx])
    return labels, targets


def sample_minibatch(labels, num_pos=32, batch_size=128):
    """양성 최대 32개 + 나머지를 배경으로 채워 128개 미니배치를 만든다."""
    pos = (labels > 0).nonzero().flatten()
    neg = (labels == 0).nonzero().flatten()
    pos = pos[torch.randperm(len(pos))[:num_pos]]
    neg = neg[torch.randperm(len(neg))[: batch_size - len(pos)]]
    return torch.cat([pos, neg])


def make_dummy_sample(num_props=300):
    H, W = 300, 400
    image = torch.randn(3, H, W)
    gt_boxes = torch.tensor([[60.0, 50.0, 200.0, 180.0], [220.0, 110.0, 380.0, 280.0]])
    gt_labels = torch.tensor([3, 7])
    # GT 주변에 지터를 준 제안 + 무작위 제안
    jitter = gt_boxes[torch.randint(0, 2, (num_props,))] + torch.randn(num_props, 4) * 15
    rand_xy = torch.rand(num_props, 2) * torch.tensor([W - 60.0, H - 60.0])
    rand_wh = 30 + torch.rand(num_props, 2) * 120
    rand = torch.cat([rand_xy, rand_xy + rand_wh], dim=1)
    proposals = torch.cat([jitter[: num_props // 2], rand[: num_props // 2]])
    proposals[:, 0::2] = proposals[:, 0::2].clamp(0, W - 1)
    proposals[:, 1::2] = proposals[:, 1::2].clamp(0, H - 1)
    keep = ((proposals[:, 2] - proposals[:, 0]) > 8) & ((proposals[:, 3] - proposals[:, 1]) > 8)
    return image, proposals[keep], gt_boxes, gt_labels


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    image, proposals, gt_boxes, gt_labels = make_dummy_sample()
    labels, targets = label_proposals(proposals, gt_boxes, gt_labels)
    print(f"proposals={len(proposals)}  positives={(labels > 0).sum().item()}  background={(labels == 0).sum().item()}")

    idx = sample_minibatch(labels)
    patches = torch.stack([warp_region(image, proposals[i]) for i in idx]).to(device)

    model.train()
    for step in range(1, 4):
        cls_logits, box_deltas = model(patches)
        loss, cls_l, reg_l = rcnn_loss(cls_logits, box_deltas, labels[idx].to(device), targets[idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[step {step}] loss={loss.item():.4f} cls={cls_l.item():.4f} reg={reg_l.item():.4f}")


if __name__ == "__main__":
    main()
