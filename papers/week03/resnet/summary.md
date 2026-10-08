# Week 3-2. ResNet

**Deep Residual Learning for Image Recognition**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun — *CVPR*, 2016 (arXiv 2015)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

- VGG, GoogLeNet 이후 "깊을수록 좋다"는 믿음이 강해졌지만, 단순히 층을 쌓는다고 항상 좋아지지는 않았다.
- 그래디언트 소실/폭발은 BatchNorm과 적절한 초기화로 상당 부분 해결되었다(BN 논문 참고).
- 그런데도 **degradation 문제**가 남는다: 층이 더 깊은 plain 망(20층 vs 56층)이 **학습 오차(training error)조차 더 높다**.
  따라서 과적합이 아니라 **최적화의 어려움** 문제이다.
- 논리적으로 깊은 망은 얕은 망 + 항등 매핑(identity) 층들로 최소한 같은 성능을 낼 수 있어야 하지만, solver가 이를 찾지 못한다.

## 2. 핵심 아이디어: 잔차 학습 (Residual Learning)

- 원하는 매핑을 H(x)라 할 때, 층들이 H(x)를 직접 학습하는 대신 **잔차 F(x) = H(x) − x** 를 학습하도록 하고 출력은 `F(x) + x`로 구성한다.
- 항등 매핑이 최적에 가깝다면 F → 0 으로 가중치를 0 쪽으로 밀기만 하면 되므로 최적화가 쉽다는 가설이다.
- 구현: **shortcut(skip) connection**으로 입력을 더한다. 추가 파라미터와 연산량이 없다(차원이 같을 때).
- 역전파 관점: y = x + F(x) 이므로 ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x). 항등 경로 덕분에 그래디언트가 깊은 층까지 직접 전달된다.

## 3. 아키텍처 / 방법

### 3.1 빌딩 블록

| 블록 | 구성 | 사용처 |
|---|---|---|
| BasicBlock | 3×3 conv–BN–ReLU → 3×3 conv–BN, + shortcut, ReLU | ResNet-18/34 |
| Bottleneck | 1×1(축소) → 3×3 → 1×1(확장 ×4), 각각 BN, + shortcut, ReLU | ResNet-50/101/152 |

수식: `y = F(x, {Wᵢ}) + x` (차원이 다르면 `y = F(x, {Wᵢ}) + Wₛx`, Wₛ는 1×1 conv로 투영).

- 차원이 증가하는 shortcut의 두 옵션: (A) 0 패딩 항등, (B) 1×1 conv 투영. 논문은 (B)가 약간 더 좋지만 핵심은 아니라고 밝히고,
  깊은 모델(50+)에서는 차원이 바뀌는 곳만 투영(B)을 사용했다.
- 해상도를 줄일 때는 stride 2를 해당 stage의 첫 블록에서 사용한다. 채널은 stage마다 2배.
- Bottleneck으로 깊이를 늘려도 연산량을 억제한다 (ResNet-50 약 3.8 GFLOPs, ResNet-152 약 11.3 GFLOPs; 논문 기준 곱셈-덧셈 횟수).

### 3.2 전체 구조 (ImageNet, 입력 224×224)

| Stage | 출력 크기 | 18-layer | 34-layer | 50-layer | 101-layer | 152-layer |
|---|---|---|---|---|---|---|
| conv1 | 112×112 | 7×7, 64, stride 2 | ← 동일 | ← | ← | ← |
| (pool) | 56×56 | 3×3 maxpool, stride 2 | | | | |
| conv2_x | 56×56 | Basic ×2 | Basic ×3 | Bottle ×3 | ×3 | ×3 |
| conv3_x | 28×28 | Basic ×2 | Basic ×4 | Bottle ×4 | ×4 | ×8 |
| conv4_x | 14×14 | Basic ×2 | Basic ×6 | Bottle ×6 | ×23 | ×36 |
| conv5_x | 7×7 | Basic ×2 | Basic ×3 | Bottle ×3 | ×3 | ×3 |
| head | 1×1 | global average pool → FC 1000 | | | | |

- FC 층을 크게 쓰지 않고 global average pooling을 써서 파라미터를 줄였다 (ResNet-50 약 2,560만 개, ResNet-152 약 6,000만 개 파라미터 수준).
- 모든 conv 뒤에 BN, Dropout 미사용.
- CIFAR-10 실험에서는 3×3 conv 스택 6n+2층(n=3,5,7,9,18…, 20/32/44/56/110층)과 1202층 모델도 실험.

### 3.3 학습 설정 (ImageNet)

- SGD, momentum 0.9, weight decay 1e-4, 배치 256, 초기 lr 0.1 (plateau에서 10배씩 감소), 약 60만 iteration 규모.
- 스케일 증강(짧은 변 256–480 랜덤 리사이즈 후 224 crop), 좌우 반전, 색 증강. 가중치 초기화는 He 초기화(Kaiming init)를 사용.

## 4. 실험 결과 요약

- **Plain vs ResNet (ImageNet, 18/34층)**: plain-34는 plain-18보다 학습/검증 오차가 모두 높음(degradation).
  ResNet-34는 ResNet-18보다 확연히 좋고 plain-34보다도 좋음 → 깊이의 이득을 실제로 얻음.
- **깊이 확장**: 50 → 101 → 152층으로 갈수록 정확도가 계속 향상(single-model top-5 오류율은 ResNet-152가 약 5.7%).
- **ILSVRC 2015**: 앙상블로 top-5 테스트 오류율 약 3.57%로 분류 부문 1위. 같은 대회의 ImageNet 검출/위치추정, COCO 검출/분할 대회에서도 모두 1위.
- **CIFAR-10**: 110층 ResNet 오류율 약 6.4%. 1202층도 학습은 되지만 과적합으로 110층보다 약간 나빠짐.
- **잔차 응답 분석**: ResNet의 층 출력은 plain 대비 작은 값을 가져, 잔차 함수가 0에 가깝다는 가설과 부합.

## 5. 의의

- 앞선 논문과의 연결: VGG의 단순한 3×3 스택 철학을 이어받으면서(plain 기준 구조가 VGG-19에서 영감), GoogLeNet보다 훨씬 단순한 모듈로 100층 이상을 학습.
  BatchNorm이 그래디언트 소실을 막아 주고, skip connection이 degradation 최적화 문제를 푼다.
- 뒤 논문과의 연결: DenseNet(다음 주, concat 기반 skip), 검출/분할의 백본(Faster R-CNN, FPN, Mask R-CNN 등), Transformer의 residual 구조,
  그리고 identity mapping 순서를 바꾼 pre-activation ResNet(He et al., 2016), ResNeXt, ResNet-D/RS 등 다양한 변형의 기반.
- 한 문장: "스킵 연결 덕분에 깊이가 곧 성능이 되었다."

## 6. 구현 노트

- `implementation/model.py`: `BasicBlock`, `Bottleneck`, 일반 `ResNet` 클래스를 처음부터 구현하고 `resnet18/34/50/101/152` 팩토리를 제공.
  `residual=False`로 shortcut을 끄면 plain 망이 되어 degradation 비교 실험이 가능하다. `cifar_stem=True`면 3×3 stem, maxpool 없음(32×32용).
- 마지막 BN의 γ를 0으로 초기화하는 `zero_init_residual` 옵션 제공 (Goyal et al. 2017 등에서 쓰는 현대적 트릭, 원 논문에는 없음).
- 투영 shortcut은 option B(1×1 conv + BN)만 구현.
- `implementation/train.py`: CIFAR-10에서 plain-N vs ResNet-N(기본 18층 규모)을 비교한다.
