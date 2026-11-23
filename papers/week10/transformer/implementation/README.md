# Transformer 구현

`model.py`: Multi-head attention, sinusoidal positional encoding, Encoder/Decoder, causal mask, greedy decoding 을
`nn.MultiheadAttention` 없이 처음부터 구현한 PyTorch 모듈.
`train.py`: 외부 데이터 없이 합성 "시퀀스 뒤집기" 과제로 Noam warmup LR + label smoothing 학습을 시연.

## 실행 방법

```bash
pip install torch
python train.py --steps 1500        # 데이터 다운로드 불필요
python train.py --pre_ln            # Pre-LN 변형
```

## 구조 확인만 하고 싶다면

```bash
python model.py   # 출력 shape, causal mask 검증, base 설정 파라미터 수
```
