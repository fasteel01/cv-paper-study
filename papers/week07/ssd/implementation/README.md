# SSD 구현

`model.py`: VGG-16 기반 SSD300 (dilated fc6, L2Norm, extras, 멀티스케일 head), default box 생성, 매칭, MultiBox 손실(hard negative mining), 디코딩 + 클래스별 NMS.
`train.py`: 더미 데이터로 손실 계산과 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
