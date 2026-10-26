"""
YOLOv1 학습 step 데모 (더미 데이터).

PASCAL VOC는 무겁기 때문에 더미 이미지와 GT 박스로
  1) GT 박스 -> 7x7x25 격자 타깃 인코딩
  2) 448x448 입력 forward -> 7x7x30 출력
  3) sum-squared error 멀티파트 손실 (좌표 5배, no-object 0.5배) 계산 및 역전파
  4) 디코딩 + NMS
를 보여준다. 손실이 같은 더미 배치에서 줄어드는지(과적합 가능 여부) 확인하는 용도.

실행:
    python train.py
"""

import torch

from model import YOLOv1, decode_predictions, encode_target, yolo_loss


def make_dummy_batch(n=2):
    images = torch.randn(n, 3, 448, 448)
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.2, 0.3, 0.2, 0.2]])  # 정규화 (cx, cy, w, h)
    labels = torch.tensor([14, 6])
    targets = torch.stack([encode_target(boxes, labels) for _ in range(n)])
    return images, targets


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    # 논문은 초기 epoch에서 lr을 서서히 올린 뒤 낮추는 스케줄(momentum 0.9, wd 5e-4)을 쓴다. 데모에서는 작은 고정 lr.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    images, targets = make_dummy_batch()
    images, targets = images.to(device), targets.to(device)

    model.train()
    for step in range(1, 4):
        pred = model(images)
        loss, parts = yolo_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[step {step}] loss={loss.item():.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))

    model.eval()
    with torch.no_grad():
        dets = decode_predictions(model(images).cpu(), conf_thresh=0.05)
    print("detections (image 0):", tuple(dets[0][0].shape))


if __name__ == "__main__":
    main()
