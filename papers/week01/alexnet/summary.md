# Week 1-2. AlexNet

**ImageNet Classification with Deep Convolutional Neural Networks**
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton — *NeurIPS*, 2012
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

2012년 이전까지 ImageNet(1000개 클래스, 120만+ 이미지) 같은 대규모 이미지 분류 문제에서
CNN은 계산량과 과적합 문제로 널리 쓰이지 못했다. AlexNet은 다음을 통해 이를 해결하고
ILSVRC-2012에서 top-5 오류율 15.3%로 2위(26.2%)를 압도적으로 이기며 딥러닝 붐을 촉발했다.

- GPU 2장을 이용한 병렬 학습으로 큰 모델을 현실적인 시간에 학습
- 여러 정규화·최적화 기법으로 대규모 망의 과적합 억제

## 2. 핵심 아이디어

1. **ReLU 활성화 함수**: `tanh`/`sigmoid` 대비 그래디언트 소실이 적고 학습이 훨씬 빠름(논문에서 6배 가량 빠른 수렴 보고).
2. **GPU 2개 병렬 학습**: 모델을 두 GPU에 나눠 일부 레이어에서만 서로 통신 → 메모리 제약 극복 (오늘날 구현은 보통 단일 GPU로 통합).
3. **Local Response Normalization(LRN)**: 인접 채널 간 경쟁을 유도하는 정규화(현재는 BatchNorm 등으로 대체되어 잘 쓰이지 않음).
4. **Overlapping Max Pooling**: 풀링 윈도우를 겹치게 하여(stride < kernel size) 약간의 성능 향상.
5. **Dropout**: FC층에서 뉴런을 확률적으로 비활성화하여 과적합 억제(당시로서는 신선한 규제 기법).
6. **데이터 증강(Data Augmentation)**: 랜덤 크롭/좌우 반전, PCA 기반 색상 변형(fancy PCA)으로 데이터 다양성 확보.

## 3. 아키텍처 개요

입력: 224×224×3 (RGB) 이미지

| Layer | 종류 | 커널/스트라이드 | 출력 채널 |
|---|---|---|---|
| Conv1 | Convolution + ReLU + LRN + MaxPool | 11×11, stride 4 | 96 |
| Conv2 | Convolution + ReLU + LRN + MaxPool | 5×5, pad 2 | 256 |
| Conv3 | Convolution + ReLU | 3×3, pad 1 | 384 |
| Conv4 | Convolution + ReLU | 3×3, pad 1 | 384 |
| Conv5 | Convolution + ReLU + MaxPool | 3×3, pad 1 | 256 |
| FC6 | Fully Connected + ReLU + Dropout | - | 4096 |
| FC7 | Fully Connected + ReLU + Dropout | - | 4096 |
| FC8 | Fully Connected (출력) | - | 1000 |

총 8개 학습 가능 레이어(conv 5개 + fc 3개), 약 6천만 개 파라미터.

## 4. 의의

- "깊은 CNN + 큰 데이터 + GPU"의 조합이 실제로 통한다는 것을 보여준 전환점.
- ReLU, Dropout, 데이터 증강 등 오늘날까지 표준으로 쓰이는 기법들을 대중화.
- 이후 VGG, GoogLeNet, ResNet 등 ImageNet 경쟁의 시대를 염.

## 5. 구현 노트

`implementation/model.py`에서 단일 GPU 기준으로 통합한 AlexNet 구조를 구현했다(원 논문의 2-GPU 분할 구조는 생략,
현대 구현에서 일반적인 방식). LRN은 옵션으로 유지하되 기본값은 비활성화했다.
