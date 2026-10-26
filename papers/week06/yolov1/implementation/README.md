# YOLOv1 구현

`model.py`: 24 conv + 2 FC 구조의 YOLOv1(7×7×30 출력), 타깃 인코딩, 논문의 sum-squared error 멀티파트 손실, 디코딩 + NMS 후처리.
`train.py`: 더미 이미지/박스로 타깃 인코딩 → forward → 손실 → 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
