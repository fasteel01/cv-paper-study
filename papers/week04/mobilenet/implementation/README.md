# MobileNet 구현

`model.py`: depthwise separable conv 기반 MobileNet v1을 처음부터 구현. `width_mult`(α)와 `separable=False`(표준 conv 비교용) 지원.
`train.py`: CIFAR-10에서 depthwise separable 버전과 표준 conv 버전을 같은 설정으로 학습해 파라미터 수·정확도를 비교.

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 10
python train.py --epochs 10 --width-mult 0.5
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
