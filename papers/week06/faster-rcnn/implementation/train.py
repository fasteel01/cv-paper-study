"""
Faster R-CNN 학습 step 데모 (더미 데이터).

PASCAL VOC / COCO는 무겁기 때문에 더미 이미지와 GT 박스로
  1) 백본 conv 특징 계산 (RPN과 검출 헤드가 공유)
  2) 앵커 라벨링/샘플링 -> RPN 손실 (objectness CE + smooth L1)
  3) RPN 제안(NMS) -> RoI 샘플링 -> 검출 헤드 손실 (CE + smooth L1)
  4) 4개 손실의 합으로 한 번에 역전파 (근사 joint 학습)
을 보여준다.

실행:
    python train.py
"""

import torch

from model import FasterRCNN

NUM_CLASSES = 20


def make_dummy_batch(n=2, h=256, w=320):
    images = torch.randn(n, 3, h, w)
    gt_boxes = [torch.tensor([[40.0, 30.0, 160.0, 150.0], [180.0, 90.0, 300.0, 230.0]]) for _ in range(n)]
    gt_labels = [torch.tensor([5, 12]) for _ in range(n)]
    return images, gt_boxes, gt_labels


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FasterRCNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    model.train()
    for step in range(1, 4):
        images, gt_boxes, gt_labels = make_dummy_batch()
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]

        losses = model(images, gt_boxes, gt_labels)
        total = sum(losses.values())
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        print(f"[step {step}] total={total.item():.4f} " + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))


if __name__ == "__main__":
    main()
