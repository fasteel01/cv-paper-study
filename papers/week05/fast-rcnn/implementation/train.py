"""
Fast R-CNN 학습 step 데모 (더미 데이터).

PASCAL VOC는 무겁고 Selective Search가 필요하므로, 더미 이미지/GT/제안 박스로
  1) 이미지당 RoI 샘플링 (64개: 양성 25% / 배경 IoU [0.1, 0.5))
  2) 이미지 2장 미니배치 forward (conv는 이미지당 1회)
  3) multi-task loss (cross-entropy + smooth L1) 계산 및 역전파
를 보여준다.

실행:
    python train.py
"""

import torch

from model import FastRCNN, fast_rcnn_loss, sample_rois

NUM_CLASSES = 20


def make_dummy_image(h=256, w=320, num_props=400):
    image = torch.randn(3, h, w)
    gt_boxes = torch.tensor([[40.0, 30.0, 160.0, 150.0], [180.0, 90.0, 300.0, 230.0]])
    gt_labels = torch.tensor([5, 12])
    jitter = gt_boxes[torch.randint(0, 2, (num_props // 2,))] + torch.randn(num_props // 2, 4) * 12
    xy = torch.rand(num_props // 2, 2) * torch.tensor([w - 60.0, h - 60.0])
    wh = 30 + torch.rand(num_props // 2, 2) * 120
    props = torch.cat([jitter, torch.cat([xy, xy + wh], dim=1)])
    props[:, 0::2] = props[:, 0::2].clamp(0, w - 1)
    props[:, 1::2] = props[:, 1::2].clamp(0, h - 1)
    ok = ((props[:, 2] - props[:, 0]) > 8) & ((props[:, 3] - props[:, 1]) > 8)
    return image, props[ok], gt_boxes, gt_labels


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastRCNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    model.train()
    for step in range(1, 4):
        images, rois, labels, targets = [], [], [], []
        for b in range(2):  # N = 2 이미지
            img, props, gt_boxes, gt_labels = make_dummy_image()
            p, l, t = sample_rois(props, gt_boxes, gt_labels, rois_per_image=64)
            images.append(img)
            rois.append(torch.cat([torch.full((len(p), 1), float(b)), p], dim=1))
            labels.append(l)
            targets.append(t)
        images = torch.stack(images).to(device)
        rois, labels, targets = torch.cat(rois).to(device), torch.cat(labels).to(device), torch.cat(targets).to(device)

        cls_logits, box_deltas = model(images, rois)
        loss, cls_l, loc_l = fast_rcnn_loss(cls_logits, box_deltas, labels, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"[step {step}] rois={len(rois)} pos={(labels > 0).sum().item()} "
            f"loss={loss.item():.4f} cls={cls_l.item():.4f} loc={loc_l.item():.4f}"
        )


if __name__ == "__main__":
    main()
