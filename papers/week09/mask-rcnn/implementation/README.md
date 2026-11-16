# Mask R-CNN 구현

`model.py`: ResNet-50 + FPN(P2~P6), RPN(앵커 · 매칭 · 샘플링 · 손실), 다중 스케일 RoIAlign(`torchvision.ops.roi_align` 사용),
Box head(FC1024×2, 클래스별 박스 회귀), **Mask head(conv×4 → deconv → 클래스별 28×28 마스크, 정답 클래스 채널만 sigmoid BCE)**,
추론 시 NMS 후 마스크를 원본 크기에 붙이는 `paste_mask`까지 직접 구현. NMS/RoIAlign 연산 자체만 torchvision 사용.
`train.py`: 합성 데이터(사각형/원 인스턴스)로 5개 손실을 합쳐 학습하는 루프 + 추론/마스크 IoU 계산 (데이터셋 다운로드 없음).

## 실행 방법

```bash
pip install torch torchvision
python train.py
```

## 구조 확인만 하고 싶다면

```bash
python model.py
```
