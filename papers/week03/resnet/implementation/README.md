# ResNet 구현

`model.py`: `BasicBlock`/`Bottleneck`/`ResNet`을 처음부터 구현, ResNet-18/34/50/101/152 제공. `residual=False`로 plain 망 비교 가능.
`train.py`: CIFAR-10에서 같은 깊이의 plain vs ResNet을 학습해 비교 (degradation 확인).

## 실행 방법

```bash
pip install torch torchvision
python train.py --arch resnet18 --epochs 10
python train.py --arch resnet34 --epochs 10   # 깊을수록 plain과의 차이가 커지는 경향
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
