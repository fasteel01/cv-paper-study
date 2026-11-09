"""
FCN 학습 step 데모 (더미 데이터).

PASCAL VOC / Cityscapes 같은 분할 데이터셋은 무겁기 때문에, 간단한 합성 데이터(배경 위의 사각형/원)로
  1) 픽셀 단위 교차 엔트로피 (ignore_index=255 지원)
  2) 몇 step 학습 후 loss 감소 확인
  3) 평가 지표: 픽셀 정확도, mean IoU (혼동 행렬 기반)
를 보여준다. VGG-16 전체(약 1.3억 파라미터)는 CPU 에서 무거우므로 fc_channels 를 줄인 FCN 을 사용한다.
(실제 학습은 FCN(num_classes=21, fc_channels=4096) + VOC 로 확장)

실행:
    python train.py
"""

import torch
import torch.nn.functional as F

from model import FCN

NUM_CLASSES = 3  # 0 = 배경, 1 = 사각형, 2 = 원


def make_synthetic_batch(n=4, size=64):
    """랜덤 위치의 사각형(클래스 1)과 원(클래스 2)이 있는 이미지. 색은 클래스와 약하게만 상관."""
    images = torch.rand(n, 3, size, size) * 0.2
    masks = torch.zeros(n, size, size, dtype=torch.long)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for i in range(n):
        x0, y0 = torch.randint(4, size // 2, (2,)).tolist()
        w, h = torch.randint(10, size // 2 - 2, (2,)).tolist()
        masks[i, y0 : y0 + h, x0 : x0 + w] = 1
        images[i, 0, y0 : y0 + h, x0 : x0 + w] += 0.7
        cx, cy = torch.randint(size // 2, size - 8, (2,)).tolist()
        r = int(torch.randint(5, 10, (1,)))
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        masks[i][circle] = 2
        images[i, 1][circle] += 0.7
    return images, masks


@torch.no_grad()
def confusion_matrix(pred, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    idx = num_classes * target[valid] + pred[valid]
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics(conf):
    conf = conf.float()
    pixel_acc = conf.diag().sum() / conf.sum().clamp(min=1)
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)
    return pixel_acc.item(), iou.mean().item()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN(num_classes=NUM_CLASSES, stage="8s", fc_channels=256).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    for step in range(1, 31):
        model.train()
        images, masks = make_synthetic_batch()
        images, masks = images.to(device), masks.to(device)
        logits = model(images)  # (N, C, H, W)
        loss = F.cross_entropy(logits, masks, ignore_index=255)  # 모든 픽셀의 CE 평균
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == 1:
            print(f"[step {step:2d}] loss={loss.item():.4f}")

    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(5):
            images, masks = make_synthetic_batch()
            pred = model(images.to(device)).argmax(1)
            conf += confusion_matrix(pred, masks.to(device), NUM_CLASSES)
    acc, miou = metrics(conf)
    print(f"eval (합성 데이터): pixel acc={acc:.3f} | mean IoU={miou:.3f}")


if __name__ == "__main__":
    main()
