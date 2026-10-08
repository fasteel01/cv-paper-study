# VGGNet 구현

`model.py`: 구성 A/B/D/E(VGG-11/13/16/19)를 처음부터 구현한 PyTorch `nn.Module` (옵션: BatchNorm).
`train.py`: CIFAR-10에서 VGG-11을 BatchNorm 없음/있음으로 비교하는 데모 스크립트.

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 5
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
