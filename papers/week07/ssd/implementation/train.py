"""
SSD300 학습 step 데모 (더미 데이터).

PASCAL VOC / COCO 는 무겁기 때문에, 무작위 이미지와 GT 박스로
  1) forward (8732개 default box 별 loc / conf 예측)
  2) GT <-> default box 매칭 + hard negative mining 이 포함된 MultiBox 손실
  3) SGD 로 역전파
를 보여준다. (손실이 줄어드는지 보려면 같은 배치를 반복해서 과적합시켜 본다.)

실행:
    python train.py
"""

import torch

from model import MultiBoxLoss, SSD300

NUM_CLASSES = 21


def make_dummy_batch(n=2, size=300):
    images = torch.randn(n, 3, size, size)
    boxes = [torch.tensor([[0.10, 0.15, 0.50, 0.60], [0.55, 0.40, 0.95, 0.90]]) for _ in range(n)]
    labels = [torch.tensor([5, 12]) for _ in range(n)]
    return images, boxes, labels


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSD300(NUM_CLASSES).to(device)
    criterion = MultiBoxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    images, boxes, labels = make_dummy_batch()
    images = images.to(device)
    boxes = [b.to(device) for b in boxes]
    labels = [l.to(device) for l in labels]

    model.train()
    for step in range(1, 6):
        loc, conf = model(images)
        loss_c, loss_l = criterion(loc, conf, boxes, labels, model.default_boxes)
        loss = loss_c + loss_l
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        print(f"[step {step}] total={loss.item():.4f} conf={loss_c.item():.4f} loc={loss_l.item():.4f}")


if __name__ == "__main__":
    main()
