# DenseNet 구현

`model.py`: `_DenseLayer`/`_DenseBlock`/`_Transition`/`DenseNet`을 처음부터 구현. DenseNet-121/169/201/161(ImageNet형)과 CIFAR형(BC-100, k=12 등) 제공.
`train.py`: CIFAR-10에서 DenseNet을 학습·평가 (`--depth`, `--growth-rate`, `--no-bc`로 변경 가능).

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 10
python train.py --depth 40 --growth-rate 12 --no-bc --epochs 10   # 비-BC 버전 (L=40, k=12)
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
