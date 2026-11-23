"""
Transformer 학습 데모: 합성 "시퀀스 뒤집기" 과제 (외부 데이터 불필요).

  입력 [a b c d] -> 출력 [d c b a]
논문의 학습 요소를 함께 보여준다: Adam(β2=0.98, eps=1e-9) + warmup LR(Noam), label smoothing, teacher forcing,
마지막에 greedy decoding 정확도 측정.

실행:
    python train.py --steps 1500
"""

import argparse

import torch
import torch.nn as nn

from model import Transformer

PAD, BOS, EOS = 0, 1, 2
FIRST = 3  # 실제 토큰은 3 이상


def make_batch(batch, length, vocab, device):
    x = torch.randint(FIRST, vocab, (batch, length), device=device)
    y = x.flip(1)
    src = x
    tgt_in = torch.cat([torch.full((batch, 1), BOS, device=device), y], dim=1)
    tgt_out = torch.cat([y, torch.full((batch, 1), EOS, device=device)], dim=1)
    return src, tgt_in, tgt_out


def noam_lr(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--pre_ln", action="store_true", help="Pre-LN 사용 (기본: 논문의 Post-LN)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(args.vocab, args.vocab, d_model=args.d_model, num_layers=2, dropout=0.1,
                        pad_idx=PAD, norm_first=args.pre_ln).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam_lr(s, args.d_model, args.warmup))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(args.batch, args.length, args.vocab, device)
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, args.vocab), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.5f}")

    model.eval()
    src, _, tgt_out = make_batch(256, args.length, args.vocab, device)
    pred = model.greedy_decode(src, BOS, args.length + 1)
    acc = (pred == tgt_out).all(dim=1).float().mean().item()
    print(f"greedy decode exact-match accuracy: {acc * 100:.1f}%")
    print("예시  src:", src[0].tolist(), "\n      pred:", pred[0].tolist())


if __name__ == "__main__":
    main()
