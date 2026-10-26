# Faster R-CNN 구현

`model.py`: 앵커 생성, Region Proposal Network(RPN), 앵커 타깃 할당/샘플링, 제안 생성(디코딩 + NMS), RoI 헤드, RPN + 검출 손실.
NMS와 RoIPool은 순수 PyTorch로 직접 구현했다(상세 차이점은 `model.py` docstring 참고).
`train.py`: 더미 데이터로 RPN 손실 + 검출 헤드 손실을 합쳐 학습 step을 보여주는 데모 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
