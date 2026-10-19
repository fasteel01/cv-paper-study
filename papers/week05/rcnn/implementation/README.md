# R-CNN 구현

`model.py`: IoU / 박스 인코딩·디코딩 / 영역 warp / NMS / AlexNet 계열 특징 추출기 / 분류·회귀 헤드 / 추론 파이프라인.
Selective Search와 클래스별 SVM은 구현하지 않았다(상세 차이점은 `model.py` docstring 참고).
`train.py`: 더미 데이터로 제안 라벨링 → 미니배치 샘플링 → 손실 계산 → 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
