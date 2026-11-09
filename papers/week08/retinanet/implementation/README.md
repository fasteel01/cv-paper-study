# RetinaNet 구현

`model.py`: ResNet-50 백본(처음부터 구현), FPN P3~P7(P6/P7은 stride-2 conv), 9개 앵커 생성, 레벨 간 가중치를 공유하는
분류/회귀 서브넷(4×conv + 출력 conv, prior π=0.01 bias 초기화), Focal Loss, 추론용 디코딩 + 클래스별 NMS.
`train.py`: 더미 데이터로 앵커 매칭(IoU 0.5/0.4) + Focal Loss + smooth L1 손실을 계산하고 학습 step을 보여주는 데모.
마지막에 Focal Loss와 BCE에서 easy negative가 손실에서 차지하는 비중을 비교한다 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch torchvision
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
