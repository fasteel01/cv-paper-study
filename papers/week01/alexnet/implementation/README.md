# AlexNet 구현

`model.py`: 논문 아키텍처를 처음부터 구현한 PyTorch `nn.Module` (단일 GPU 기준으로 통합).
`train.py`: 파이프라인 확인용 CIFAR-10 학습 스크립트 (실제 논문 재현은 ImageNet 필요).

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 3
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
