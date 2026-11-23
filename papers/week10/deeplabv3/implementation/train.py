"""
DeepLabv3 학습 데모 (더미 데이터).

PASCAL VOC 2012 / Cityscapes 는 용량이 커서 자동 다운로드하지 않는다.
여기서는 랜덤 이미지/마스크로 다음을 보여준다.
  1) 픽셀별 cross-entropy (ignore_index=255: VOC 의 경계/void 라벨)
  2) poly 학습률 스케줄: lr = base * (1 - it/max_it) ** 0.9
  3) 몇 step 학습 (같은 더미 배치를 반복 -> loss 가 줄어드는지 확인)

실행:
    python train.py --steps 5
"""

import argparse

import torch
import torch.nn.functional as F

from model import DeepLabV3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--size", type=int, default=129)
    ap.add_argument("--classes", type=int, default=21)
    ap.add_argument("--os", type=int, default=16, choices=[8, 16, 32])
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLabV3(args.classes, output_stride=args.os, multi_grid=(1, 2, 4)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # BN 때문에 batch >= 2 필요
    images = torch.randn(2, 3, args.size, args.size, device=device)
    masks = torch.randint(0, args.classes, (2, args.size, args.size), device=device)
    masks[:, :4, :] = 255  # void 영역 예시

    model.train()
    for it in range(args.steps):
        lr = args.lr * (1 - it / args.steps) ** 0.9
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(images)  # 이미 입력 해상도로 업샘플됨
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {it + 1}/{args.steps}  lr={lr:.5f}  loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
