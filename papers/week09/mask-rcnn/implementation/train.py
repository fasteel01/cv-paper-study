"""
Mask R-CNN 학습 step 데모 (합성 데이터).

COCO 는 너무 무거우므로, 배경 위의 사각형(클래스 1)/원(클래스 2)을 합성한 인스턴스 분할 데이터로
  1) 5개 손실(rpn_objectness, rpn_box, box_cls, box_reg, mask)을 계산하고 합쳐 역전파하는 학습 루프
  2) 같은 데이터에 몇 step 학습한 뒤 loss 감소 확인 (사전학습 없는 ResNet-50 이라 '과적합 데모' 수준)
  3) 추론 결과(박스/라벨/점수/마스크)와 정답 마스크의 IoU 계산 방법
을 보여준다. 수치는 논문과 비교할 수 없다. (실제 학습: COCO train2017, 짧은 변 800, SGD lr 0.02, 90k iter @ 배치 16)

실행:
    python train.py
"""

import torch

from model import MaskRCNN

NUM_FG = 2  # 1 = 사각형, 2 = 원


def make_sample(size=128):
    """이미지 (3,H,W) 와 targets dict(boxes, labels, masks). 객체 2~3개."""
    img = torch.rand(3, size, size) * 0.2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    boxes, labels, masks = [], [], []
    for _ in range(int(torch.randint(2, 4, (1,)))):
        cls = int(torch.randint(1, NUM_FG + 1, (1,)))
        r = int(torch.randint(14, 26, (1,)))
        cx, cy = torch.randint(r, size - r, (2,)).tolist()
        if cls == 1:
            m = ((xx - cx).abs() <= r) & ((yy - cy).abs() <= r)
        else:
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[cls - 1][m] += 0.7
        ys, xs = m.nonzero(as_tuple=True)
        boxes.append([xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1])
        labels.append(cls)
        masks.append(m.to(torch.uint8))
    return img, {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels), "masks": torch.stack(masks)}


def mask_iou(pred, gt):
    """pred (P,H,W) bool, gt (G,H,W) bool -> (P,G) IoU"""
    p, g = pred.flatten(1).float(), gt.flatten(1).float()
    inter = p @ g.T
    return inter / (p.sum(1)[:, None] + g.sum(1)[None] - inter).clamp(min=1)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskRCNN(num_classes=NUM_FG + 1, rpn_pre_nms_top_n=300, rpn_post_nms_top_n=100, box_batch_size=64).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    samples = [make_sample() for _ in range(2)]  # 고정된 2장에 과적합시키며 loss 감소 확인
    images = torch.stack([s[0] for s in samples]).to(device)
    targets = [{k: v.to(device) for k, v in s[1].items()} for s in samples]

    for step in range(1, 21):
        model.train()
        losses = model(images, targets)
        loss = sum(losses.values())  # L = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 5 == 0 or step == 1:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
            print(f"[step {step:2d}] total={loss.item():.3f} | {parts}")

    model.eval()
    model.score_thr = 0.3
    with torch.no_grad():
        dets = model(images)
    for i, (d, t) in enumerate(zip(dets, targets)):
        n = d["boxes"].size(0)
        msg = f"image {i}: {n} detections (GT {t['boxes'].size(0)})"
        if n:
            iou = mask_iou(d["masks"], t["masks"].bool())
            msg += f" | best mask IoU per detection: {[round(v, 2) for v in iou.max(1).values[:5].tolist()]}"
        print(msg)


if __name__ == "__main__":
    main()
