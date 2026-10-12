# Week 4-2. MobileNet (v1)

**MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**
Andrew G. Howard, Menglong Zhu, Bo Chen, et al. — *arXiv*, 2017
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

AlexNet 이후 CNN은 정확도를 위해 점점 깊고 커졌다(VGG-16은 약 1.4억 파라미터, 약 150억 multiply-add). 하지만 모바일·임베디드 기기나
로봇, 자율주행 같은 환경에서는 **작은 모델 크기와 낮은 지연시간(latency)**이 필요하다.
기존 접근은 (1) 큰 모델을 압축/양자화/증류하거나 (2) 작은 모델을 직접 설계하는 것(SqueezeNet 등)이었다.
MobileNet은 후자로서, **연산 대부분을 차지하는 표준 합성곱을 더 싼 연산으로 분해**하고, 두 개의 하이퍼파라미터로
정확도–속도 trade-off를 쉽게 조절할 수 있게 한다. 모델 크기뿐 아니라 **지연시간**을 함께 고려한 설계라는 점도 강조한다.

## 2. 핵심 아이디어

### 2.1 Depthwise Separable Convolution

표준 합성곱을 **depthwise conv + pointwise(1×1) conv** 두 단계로 분해한다.

- **Depthwise**: 입력 채널마다 하나의 `D_K×D_K` 필터를 독립적으로 적용 (공간 정보만 처리, 채널 간 섞임 없음)
- **Pointwise**: `1×1` conv로 채널을 섞어 `N`개 출력 채널 생성 (채널 정보만 처리)

입력 `D_F×D_F×M`, 출력 `D_F×D_F×N`, 커널 `D_K×D_K` (stride 1, same padding)일 때 연산량(multiply-adds):

| 연산 | 연산량 |
|---|---|
| 표준 conv | `D_K · D_K · M · N · D_F · D_F` |
| Depthwise | `D_K · D_K · M · D_F · D_F` |
| Pointwise | `M · N · D_F · D_F` |
| Depthwise separable (합) | `D_K² · M · D_F² + M · N · D_F²` |

비율:

```
(D_K² M D_F² + M N D_F²) / (D_K² M N D_F²) = 1/N + 1/D_K²
```

3×3 커널(`D_K = 3`)이면 N이 클 때 약 **8~9배** 적은 연산이다.
각 depthwise/pointwise conv 뒤에 BN과 ReLU를 둔다(논문 Figure 3).

### 2.2 Width multiplier α

모든 층의 채널 수를 `α`배로 줄인다 (입력 `M → αM`, 출력 `N → αN`). 연산량·파라미터가 대략 `α²`배로 감소. α ∈ {1, 0.75, 0.5, 0.25}.

### 2.3 Resolution multiplier ρ

입력 해상도(및 그에 따른 모든 내부 특징 맵)를 `ρ`배로 줄인다. 연산량이 약 `ρ²`배로 감소. 해상도 224, 192, 160, 128에 해당.
파라미터 수는 변하지 않는다.

결합 시 한 층의 연산량: `D_K² · αM · ρD_F² + αM · αN · ρD_F²`.

## 3. 아키텍처

첫 층만 표준 3×3 conv(stride 2)이고, 이후는 모두 depthwise separable conv. 총 28층(depthwise와 pointwise를 각각 센 경우),
마지막은 global avg pool → FC → softmax. 다운샘플링은 풀링이 아닌 **stride 2 depthwise conv**로 수행.

입력 224×224×3 기준:

| 층 (Type / Stride) | 필터 형태 | 출력 크기 |
|---|---|---|
| Conv / s2 | 3×3×3×32 | 112×112×32 |
| Conv dw / s1 → Conv pw / s1 | 3×3 dw → 1×1×32×64 | 112×112×64 |
| Conv dw / s2 → pw / s1 | 3×3 dw → 1×1×64×128 | 56×56×128 |
| Conv dw / s1 → pw / s1 | 3×3 dw → 1×1×128×128 | 56×56×128 |
| Conv dw / s2 → pw / s1 | 3×3 dw → 1×1×128×256 | 28×28×256 |
| Conv dw / s1 → pw / s1 | 3×3 dw → 1×1×256×256 | 28×28×256 |
| Conv dw / s2 → pw / s1 | 3×3 dw → 1×1×256×512 | 14×14×512 |
| 5× (dw / s1 → pw / s1) | 3×3 dw → 1×1×512×512 | 14×14×512 |
| Conv dw / s2 → pw / s1 | 3×3 dw → 1×1×512×1024 | 7×7×1024 |
| Conv dw / s1 → pw / s1 | 3×3 dw → 1×1×1024×1024 | 7×7×1024 |
| Avg Pool 7×7 | - | 1×1×1024 |
| FC / softmax | 1024×1000 | 1000 |

- **연산 분포**: multiply-add의 약 95%, 파라미터의 약 75%가 1×1 pointwise conv에 있다. 1×1 conv는 `im2col` 없이 GEMM으로
  바로 구현 가능해 효율적이다.
- 학습 시 depthwise에는 weight decay를 거의 주지 않는 것이 좋다고 언급(파라미터가 매우 적어 정규화가 불필요/해로움).
  또한 작은 모델이라 과적합이 적어 정규화·증강(크롭 등)을 약하게 사용한다. 최적화는 RMSprop 계열(비동기 학습)을 사용했다고 보고.

## 4. 실험 결과 요약

- **Depthwise separable vs 표준 conv** (ImageNet): 정확도 약 1%p 감소(약 70.6% vs 71.7%)에 비해 연산량·파라미터는 약 1/8~1/9 수준
  (약 5.7억 vs 48억 mult-adds, 약 4.2M vs 29.3M 파라미터).
- **얇게 vs 얕게**: 같은 연산량이면 층을 빼서 얕게 만드는 것보다 α로 얇게 만드는 편이 정확도가 더 높음.
- **ImageNet 비교**: MobileNet(1.0, 224)은 top-1 약 70.6%로, VGG-16(약 71.5%, 약 138M 파라미터, 약 15.3B mult-adds)에 근접하면서
  훨씬 작고(약 4.2M, 약 569M mult-adds), GoogLeNet(약 69.8%, 약 6.8M 파라미터, 약 1.5B mult-adds)보다 작고 정확하다.
- **α, ρ 영향**: α와 ρ를 줄이면 정확도가 완만하게 하락(예: α=0.25, 해상도 128 같은 극단 설정에서는 크게 하락). 연산량과 정확도가 매끄러운 trade-off를 형성.
- **다른 작업**: 세부 분류(Stanford Dogs), 지리 위치 추정(PlaNet 대체), 얼굴 속성 분류(증류), Face embedding(FaceNet 증류),
  객체 탐지(SSD/Faster R-CNN의 backbone, COCO) 등 분류 이상의 응용에서도 경쟁력을 보임.

> 정확한 표의 수치는 원문 Table 4~10을 참고할 것. 위 수치는 대략적인 값(약)이다.

## 5. 의의 (앞뒤 논문과의 연결)

- **효율적 설계의 출발점**: 이후 MobileNetV2(inverted residual + linear bottleneck), V3(NAS + SE + h-swish), ShuffleNet, EfficientNet이 이어진다.
- **VGG/ResNet(앞 주차)과 대비**: "더 깊고 크게" 대신 연산 분해로 같은 표현력을 더 싸게. 정확도 최대화가 아니라 **정확도/연산량의 파레토 최적**을 목표로 한다.
- **GoogLeNet의 1×1 conv 아이디어**: 1×1 conv로 채널 혼합을 담당하는 구조를 극단화(Xception의 depthwise separable과 같은 맥락).
- **다음 주차(탐지)와의 연결**: SSD, Faster R-CNN 등의 backbone으로 흔히 교체되어 쓰이며, 실시간 탐지·모바일 배포의 기초가 된다.
- **한계**: depthwise conv는 FLOPs가 적어도 메모리 접근 비율이 높아 하드웨어에서 FLOPs만큼 빠르지 않을 수 있다.
  또한 단순 직렬 구조라 residual이 없고, 낮은 α에서는 ReLU로 인한 정보 손실 문제가 있다(V2에서 해결).

## 6. 구현 노트

`implementation/model.py`:
- `DepthwiseSeparableConv`: `Conv2d(groups=in_channels)` + BN + ReLU → 1×1 `Conv2d` + BN + ReLU.
- `StandardConvBlock`: 비교용 3×3 표준 conv + BN + ReLU (`separable=False`로 선택).
- `MobileNetV1(num_classes, width_mult, separable)`: 표의 구조를 그대로 구현. `width_mult`로 α를 적용, 채널 수는 반올림(최소 8).
- 단순화: 원 논문의 ReLU는 그대로 ReLU를 사용(MobileNetV2/구현체에서 쓰는 ReLU6 대신), 해상도 배율 ρ는 입력 크기를 바꾸는 것으로 대체.
- 마지막 FC 앞 dropout은 옵션(기본 0.2)으로 제공(논문 고유 설정이 아닌 구현 편의).
