# FCN 구현

`model.py`: VGG-16 백본(처음부터 구현), FC → conv 변환(fc6 = 7×7 conv, fc7 = 1×1 conv), 1×1 score 층,
bilinear 초기화된 전치 합성곱 업샘플, FCN-32s / 16s / 8s skip 구조 (`stage` 인자로 선택).
`train.py`: 합성 데이터(사각형/원)로 픽셀 단위 교차 엔트로피 학습 + 픽셀 정확도 / mean IoU 평가 (데이터셋 다운로드 없음, CPU에서 동작하도록 fc 채널을 줄임).

## 실행 방법

```bash
pip install torch torchvision
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
