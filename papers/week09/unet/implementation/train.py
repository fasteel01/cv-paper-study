"""
U-Net 학습 데모 (합성 "세포" 데이터).

ISBI 2012 EM / PhC-U373 등 원 논문 데이터는 다운로드가 필요하므로, 여기서는 밝은 배경 위의 어두운 원형 세포를
합성해서 다음을 보여준다.
  1) 논문의 **가중치 맵 손실**: w(x) = w_c(x) + w0 * exp(-(d1(x)+d2(x))^2 / (2*sigma^2))
     (붙어 있는 세포 사이의 얇은 배경 경계에 큰 가중치)
  2) valid-convolution U-Net: 입력 > 출력 이므로 정답/가중치 맵을 중앙 crop 해서 비교
  3) 몇 step 학습 후 loss 감소와 foreground IoU 확인

실행:
    python train.py            # 기본: valid conv (논문 방식)
    python train.py --same     # same padding (입출력 크기 동일)
"""

import argparse

import torch
import torch.nn.functional as F

from model import UNet, center_crop


def make_cells(size=96, n_cells=8):
    """(image(1,H,W), instance_map(H,W; 0=배경, 1..n=세포 id)). 세포끼리 일부러 가깝게 배치."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    inst = torch.zeros(size, size, dtype=torch.long)
    for k in range(1, n_cells + 1):
        r = int(torch.randint(7, 12, (1,)))
        cx, cy = torch.randint(r, size - r, (2,)).tolist()
        inst[((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & (inst == 0)] = k
    image = 0.8 - 0.5 * (inst > 0).float() + 0.1 * torch.randn(size, size)
    return image[None], inst


def weight_map(inst, w0=10.0, sigma=5.0):
    """
    논문 식 (2). d1, d2 = 가장 가까운 세포 / 두 번째로 가까운 세포까지의 거리.
    (원 구현은 세포 '경계'까지의 거리 변환을 쓰며, 여기서는 세포 픽셀 집합까지의 최단 거리로 근사 -> 배경 픽셀에서 동일)
    w_c = 클래스 빈도 균형 가중치(빈도의 역수를 평균 1 로 정규화).
    """
    h, w = inst.shape
    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2).float()
    ids = [k for k in inst.unique().tolist() if k != 0]
    fg = (inst > 0).float()
    freq = torch.stack([1 - fg.mean(), fg.mean()]).clamp(min=1e-6)
    wc_table = (1 / freq) / (1 / freq).mean()
    wc = wc_table[(inst > 0).long()]
    if len(ids) < 2:
        return wc
    dists = torch.stack([torch.cdist(coords, coords[(inst == k).reshape(-1)]).min(1).values for k in ids], 0)
    d1, d2 = dists.topk(2, dim=0, largest=False).values
    return wc + w0 * torch.exp(-((d1 + d2) ** 2) / (2 * sigma**2)).reshape(h, w)


def make_batch(n, size):
    imgs, labels, weights = [], [], []
    for _ in range(n):
        img, inst = make_cells(size)
        imgs.append(img)
        labels.append((inst > 0).long())
        weights.append(weight_map(inst))
    return torch.stack(imgs), torch.stack(labels), torch.stack(weights)


def find_valid_input(model, min_out):
    """출력이 min_out 이상이 되는 가장 작은 유효 입력 크기 (valid conv 는 입력 크기 제약이 있음)."""
    s = min_out
    while True:
        try:
            out = model.output_size(s)
            if out >= min_out:
                return s, out
        except ValueError:
            pass
        s += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same", action="store_true", help="same padding 사용 (기본은 논문의 valid conv)")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CPU 에서 빠르게 돌도록 depth=3, base=16 의 축소판 (논문: depth=4, base=64)
    model = UNet(in_channels=1, num_classes=2, base=16, depth=3, padding=args.same).to(device)
    in_size, out_size = (96, 96) if args.same else find_valid_input(model, 64)
    print(f"input {in_size}x{in_size} -> output {out_size}x{out_size} ({'same' if args.same else 'valid'})")
    # 논문: SGD, momentum 0.99 (배치가 1 장이라 큰 momentum 으로 과거 샘플을 반영)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for step in range(1, args.steps + 1):
        model.train()
        img, label, wmap = make_batch(4, in_size)
        label, wmap = center_crop(label, (out_size, out_size)), center_crop(wmap, (out_size, out_size))
        logits = model(img.to(device))
        ce = F.cross_entropy(logits, label.to(device), reduction="none")  # (N, H', W') 픽셀별 -log p_l(x)
        loss = (wmap.to(device) * ce).sum() / wmap.sum().to(device)  # 가중 평균
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(f"[step {step:3d}] weighted loss={loss.item():.4f}")

    model.eval()
    inter = union = 0
    with torch.no_grad():
        img, label, _ = make_batch(8, in_size)
        pred = model(img.to(device)).argmax(1).cpu()
        label = center_crop(label, (out_size, out_size))
        inter = ((pred == 1) & (label == 1)).sum().item()
        union = ((pred == 1) | (label == 1)).sum().item()
    print(f"eval (합성 데이터): foreground IoU={inter / max(union, 1):.3f}")


if __name__ == "__main__":
    main()
