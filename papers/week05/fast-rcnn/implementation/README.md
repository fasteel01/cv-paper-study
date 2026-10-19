# Fast R-CNN 구현

`model.py`: 순수 PyTorch RoIPool, conv 백본 + FC + 분류/회귀 형제 헤드, multi-task loss, RoI 샘플링, Truncated SVD.
Selective Search는 구현하지 않았다(상세 차이점은 `model.py` docstring 참고).
`train.py`: 더미 데이터로 RoI 샘플링 → forward → multi-task loss → 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
