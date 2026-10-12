# Week 4-1. DenseNet

**Densely Connected Convolutional Networks**
Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger — *CVPR*, 2017 (Best Paper Award)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

네트워크가 깊어질수록 입력 정보와 그래디언트가 많은 층을 거치며 "사라지는(vanish)" 문제가 생긴다.
ResNet은 `x_l = H_l(x_{l-1}) + x_{l-1}` 형태의 항등 shortcut으로, Highway Network·FractalNet·Stochastic Depth 등도
"앞쪽 층에서 뒤쪽 층으로 가는 짧은 경로"를 만들어 이 문제를 완화했다.

DenseNet은 이 관찰을 극단까지 밀어붙인다. **같은 해상도의 모든 층을 서로 직접 연결**하면 정보·그래디언트 흐름이 최대가 되고,
이전 층의 특징 맵을 재사용(feature reuse)하므로 **층당 필요한 채널 수가 적어 파라미터 효율**이 높아진다.
또한 ResNet은 층이 많을 때 일부 층이 거의 기여하지 않는다는 지적(Stochastic Depth 논문)이 있었는데,
DenseNet은 "이미 있는 특징을 다시 학습하지 않도록" 설계되어 중복이 줄어든다.

## 2. 핵심 아이디어

### 2.1 Dense connectivity

ResNet은 특징을 **더한다(sum)**: `x_l = H_l(x_{l-1}) + x_{l-1}`.
DenseNet은 이전 모든 층의 출력을 채널 축으로 **이어 붙인다(concatenate)**:

```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

- `[·]`는 채널 방향 concat. 합(sum)과 달리 각 층의 특징이 변형 없이 보존된다.
- L개 층이 있는 블록의 직접 연결 수는 `L(L+1)/2` (전통적 CNN은 L).
- `H_l`은 **BN → ReLU → 3×3 Conv**의 합성 함수 (pre-activation 순서, ResNet v2와 유사).

### 2.2 Growth rate k

각 층 `H_l`은 **k개의 특징 맵만** 새로 만든다. l번째 층의 입력 채널 수는 `k_0 + k × (l-1)` (k_0는 블록 입력 채널 수).
k가 작아도(예: 12, 32) 충분히 좋은 성능이 나온다. 각 층이 "집단 지식(collective knowledge)"에 k개씩 덧붙이고
모든 층이 그것을 읽을 수 있다는 비유로 설명한다. 네트워크 상태(global state)를 공유하므로 층마다 넓을 필요가 없다.

### 2.3 Bottleneck (DenseNet-B)

입력 채널이 계속 늘어나므로 3×3 conv 앞에 **1×1 conv(출력 4k 채널)**를 넣어 입력 차원을 줄인다.
합성 함수는 `BN-ReLU-Conv(1×1) → BN-ReLU-Conv(3×3)`.

### 2.4 Compression (DenseNet-C)와 Transition layer

풀링으로 해상도를 줄이려면 dense block 사이에 **transition layer**(BN → 1×1 Conv → 2×2 AvgPool, stride 2)를 둔다.
이때 채널 수를 `θ`배로 줄이는 것이 compression이다 (`0 < θ ≤ 1`, 논문은 θ = 0.5). B와 C를 모두 쓴 것이 **DenseNet-BC**.

## 3. 아키텍처

### 3.1 전체 구조

`입력 → 초기 Conv → [Dense Block → Transition] × (N-1) → Dense Block → BN-ReLU → Global AvgPool → FC`

### 3.2 ImageNet용 설정 (k = 32, 초기 conv 출력 2k = 64)

초기 stem: 7×7 Conv stride 2 → 3×3 MaxPool stride 2 (224 → 56). 이후 4개의 dense block.

| 모델 | Dense Block별 층 수 (block 1~4) | growth rate k |
|---|---|---|
| DenseNet-121 | 6, 12, 24, 16 | 32 |
| DenseNet-169 | 6, 12, 32, 32 | 32 |
| DenseNet-201 | 6, 12, 48, 32 | 32 |
| DenseNet-161 | 6, 12, 36, 24 | 48 |

해상도 변화 (224 입력): 56×56 → (transition) 28×28 → 14×14 → 7×7. 이름의 숫자는 가중층 수:
DenseNet-121 = 초기 conv 1 + (6+12+24+16)×2 + transition 3 + FC 1 = 121.

채널 수 예시 (DenseNet-121, θ = 0.5): 64 → block1 후 64+6×32 = 256 → transition 후 128 → block2 후 128+12×32 = 512 → 256 →
block3 후 256+24×32 = 1024 → 512 → block4 후 512+16×32 = 1024 → FC.

### 3.3 CIFAR용 설정

3개의 dense block, 각 블록 층 수 동일, 초기 conv는 3×3 (출력 16채널, BC는 2k). 블록 사이 transition, 마지막에 8×8 global avg pool.
예: DenseNet(L=40, k=12), DenseNet-BC(L=100, k=12), DenseNet-BC(L=190, k=40).
증강을 쓰지 않는 데이터셋에서는 conv 뒤에 dropout 0.2를 사용.

### 3.4 학습 설정 (논문)

SGD + Nesterov momentum 0.9, weight decay 1e-4, He 초기화.
CIFAR: 배치 64, 300 epoch, lr 0.1을 50%, 75% 시점에 ÷10.
ImageNet: 배치 256, 90 epoch, lr 0.1을 30, 60 epoch에 ÷10.

## 4. 실험 결과 요약

- **CIFAR/SVHN**: ResNet 계열보다 낮은 오류율을 더 적은 파라미터로 달성. DenseNet-BC (L=190, k=40)가
  C10+에서 약 3.5%, C100+에서 약 17.2%로 당시 최고 수준 (+는 표준 증강 사용).
  DenseNet-BC (L=100, k=12)는 약 0.8M 파라미터로 C10+ 약 4.5% 오류율로, 훨씬 큰 ResNet-1001(약 10M 파라미터)과 견줄 만한 수준.
- **파라미터 효율**: 비슷한 정확도에서 ResNet 대비 파라미터·FLOPs가 대략 절반~1/3 수준이라고 보고.
- **ImageNet**: 비슷한 정확도의 ResNet보다 파라미터와 연산량이 적은 곡선을 보임.
- **과적합 완화**: 증강 없는 C10/C100에서도 오류율이 크게 악화되지 않아 파라미터 대비 일반화가 좋음.
- **분석(Section 5)**: 후반 층들이 앞쪽 층의 특징도 실제로 사용함(가중치 heat map)을 보여, dense 연결이 실제로 활용됨을 확인.

> 정확한 표의 수치는 원문 Table 2, 3을 참고할 것. 위 수치는 대략적인 값(약)이다.

## 5. 의의 (앞뒤 논문과의 연결)

- **ResNet(Week 3)**: 더하기 shortcut → 이어붙이기(concat)로 일반화. 특징 재사용을 명시적으로 만들어 파라미터 효율을 얻음.
- **GoogLeNet/Inception(Week 2)**: 여러 경로의 특징을 concat한다는 점은 비슷하지만, DenseNet은 모든 층이 서로 연결된다.
- **BatchNorm(Week 3)**: 모든 conv 앞에 BN-ReLU를 두는 pre-activation 구조가 깊은 dense 망의 학습을 안정화.
- **단점**: concat된 특징 맵을 모두 보관해야 해서 학습 시 **GPU 메모리 사용량이 큼**
  (후속 "Memory-efficient DenseNet"은 BN/concat을 재계산하여 완화). FLOPs가 적어도 실제 속도는 느릴 수 있음.
- **이후**: 경량화(MobileNet, 다음 논문)와 세그멘테이션의 concat skip 연결(U-Net 등)에서도 특징 재사용이 핵심 주제가 된다.

## 6. 구현 노트

`implementation/model.py`:
- `_DenseLayer`: BN-ReLU-1×1Conv(4k) → BN-ReLU-3×3Conv(k) 후 입력과 `torch.cat`.
- `_DenseBlock`: 층을 쌓으며 입력 채널이 k씩 증가. `_Transition`: BN-ReLU-1×1Conv(θ배) + AvgPool 2×2.
- `DenseNet` 클래스 하나로 ImageNet형(`small_inputs=False`: 7×7 stem + maxpool)과 CIFAR형(`small_inputs=True`: 3×3 stem) 모두 지원.
- `densenet121/169/201/161`, `densenet_cifar`(BC-100 등) 팩토리 제공.
- 단순화: 메모리 절약용 checkpoint는 생략, dropout 기본 0, Kaiming normal 초기화.
