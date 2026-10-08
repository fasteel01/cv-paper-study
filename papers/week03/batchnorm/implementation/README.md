# BatchNorm 구현

`model.py`: `BatchNorm2d`를 처음부터 구현(`torch.nn.BatchNorm2d`와 출력 일치 검증 포함) + BN 유무 전환이 가능한 `SmallCNN`.
`train.py`: CIFAR-10에서 BN 적용 전/후, 학습률별(기본 0.01, 0.1) 학습 곡선 비교.

## 실행 방법

```bash
pip install torch torchvision
python train.py --epochs 5
python train.py --epochs 5 --lrs 0.01 0.1 0.5   # 학습률 바꿔보기
```

## 구조/동작 확인만 하고 싶다면

```bash
python model.py
```
