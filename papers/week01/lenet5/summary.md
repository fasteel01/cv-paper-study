# Week 1-1. LeNet-5

**Gradient-Based Learning Applied to Document Recognition**
Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner — *Proceedings of the IEEE*, 1998
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

1990년대까지 손글씨 숫자 인식 등 문서 인식 문제는 손으로 설계한 특징 추출기(hand-crafted feature extractor) +
얕은 분류기(SVM, fully-connected network 등) 조합으로 풀렸다. 이 방식은:

- 특징 설계에 도메인 지식과 많은 튜닝이 필요하고,
- 이미지의 2차원 공간 구조(픽셀의 인접 관계)를 활용하지 못하며,
- 이미지의 이동(shift)·왜곡(distortion)에 취약하다.

LeCun et al.은 "특징 추출과 분류를 함께 학습"하는 end-to-end 그래디언트 기반 학습 프레임워크로
**합성곱 신경망(Convolutional Neural Network, CNN)**을 제안하고, 이를 우편번호/수표 숫자 인식(MNIST의 원형)에 적용했다.

## 2. 핵심 아이디어

CNN은 이미지 인식에 적합하도록 다음 3가지 구조적 제약(inductive bias)을 도입한다.

1. **지역 수용 영역(Local receptive field)**: 각 뉴런은 입력 이미지의 작은 영역만 본다 → 지역적인 edge, corner 같은 저수준 특징을 감지.
2. **가중치 공유(Shared weights)**: 같은 필터(커널)를 이미지 전체에 슬라이딩하며 적용 → 파라미터 수 감소, 위치에 무관하게 같은 특징 탐지 가능(평행이동 불변성).
3. **부분 표본화(Subsampling / Pooling)**: 특징 맵의 해상도를 점진적으로 줄여 약간의 이동·왜곡에 강건해지고 연산량을 줄임.

이 세 가지가 합쳐져 합성곱층(Convolution) → 활성화 → 서브샘플링(Pooling)을 반복하는 구조가 만들어진다.

## 3. LeNet-5 아키텍처

입력: 32×32 grayscale 이미지

| Layer | 종류 | 출력 크기 | 커널/설명 |
|---|---|---|---|
| C1 | Convolution | 6@28×28 | 5×5 kernel |
| S2 | Subsampling(Avg Pool) | 6@14×14 | 2×2 |
| C3 | Convolution | 16@10×10 | 5×5 kernel |
| S4 | Subsampling(Avg Pool) | 16@5×5 | 2×2 |
| C5 | Convolution(사실상 FC) | 120 | 5×5 kernel |
| F6 | Fully Connected | 84 | - |
| Output | Fully Connected(RBF, 논문 원본) | 10 | 클래스 수 |

- 활성화 함수: 원 논문은 스케일된 tanh (`f(x) = A*tanh(S*x)`) 사용. (오늘날 구현에서는 보통 ReLU로 대체)
- 출력층: 원 논문은 Euclidean Radial Basis Function(RBF) 사용, 오늘날은 보통 Softmax + Cross-Entropy로 대체.
- 학습: 오차 역전파(Backpropagation) + 그래디언트 기반 최적화.

## 4. 의의

- CNN의 세 가지 핵심 구조(지역 연결, 가중치 공유, 풀링)를 정식화한 최초의 성공적 사례.
- 이후 AlexNet, VGG, ResNet 등 모든 CNN 계열의 원형(prototype)이 되는 구조.
- 이번 스터디에서 "기초부터" 다지기 위한 출발점으로 선택.

## 5. 구현 노트

`implementation/model.py`에 PyTorch로 LeNet-5를 처음부터 구현했다. 최신 관례에 맞춰 활성화 함수는 ReLU,
출력층은 Softmax(Cross-Entropy Loss로 대체)를 사용했고, `train.py`에서 MNIST로 학습/검증할 수 있다.
