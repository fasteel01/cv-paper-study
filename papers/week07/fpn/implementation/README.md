# FPN 구현

`model.py`: ResNet-50 백본(처음부터 구현), Feature Pyramid Network(lateral 1×1 + top-down + 3×3 smoothing, P2~P6),
피라미드 전 층에서 가중치를 공유하는 RPN head, 레벨별 앵커 생성, RoI→피라미드 레벨 할당 규칙, 멀티레벨 RoIAlign, Fast R-CNN box head.
`train.py`: 더미 데이터로 RPN 손실 + box head 손실을 계산하고 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch torchvision
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
