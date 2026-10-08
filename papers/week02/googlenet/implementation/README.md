# GoogLeNet (Inception v1) 구현

`model.py`: Inception 모듈 9개 + 보조 분류기 2개를 처음부터 구현한 PyTorch `nn.Module`.
`train.py`: 파이프라인 확인용 CIFAR-10(64×64 리사이즈) 학습 스크립트. 보조 손실(가중치 0.3) 포함.

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 3
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
