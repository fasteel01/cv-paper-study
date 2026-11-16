# U-Net 구현

`model.py`: 논문 그대로의 U-Net (valid convolution, 중앙 crop 후 skip concat, 2×2 up-convolution, 572→388).
`padding=True` / `batch_norm=True` 옵션으로 현대적 변형(same padding, BN)도 지원.
`train.py`: 합성 세포 이미지로 논문의 **가중치 맵 손실**(붙은 세포 사이 경계 강조)을 적용해 학습하고 foreground IoU 를 평가
(데이터셋 다운로드 없음, CPU 에서 동작하도록 축소판 U-Net 사용).

## 실행 방법

```bash
pip install torch torchvision
python train.py            # valid conv (논문 방식)
python train.py --same     # same padding
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
