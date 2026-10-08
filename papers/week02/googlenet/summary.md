# Week 2-2. GoogLeNet (Inception v1)

**Going Deeper with Convolutions**
Christian Szegedy, Wei Liu, Yangqing Jia, et al. — *CVPR*, 2015 (arXiv 2014)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

AlexNet 이후 성능 향상의 가장 쉬운 길은 망을 깊게(depth), 넓게(width) 키우는 것이었다. 하지만:

- 파라미터가 늘면 **과적합** 위험이 커지고 라벨링된 데이터가 더 필요하다.
- 연산량은 인접한 두 conv 층의 채널 수를 동시에 키우면 **제곱**으로 증가해 계산 자원이 낭비된다(필터가 0에 가까운 가중치를 갖게 되는 경우가 많음).
- 이상적인 해법은 **희소(sparse) 구조**이지만, 현재의 하드웨어(GPU)는 희소 행렬 연산에 비효율적이다.

저자들은 "희소 구조를 **조밀한(dense) 구성 요소들로 근사**"하는 것을 목표로 Inception 모듈을 설계했다.
대회 제출 이름은 LeNet에 대한 경의를 담은 **GoogLeNet** 이며, ILSVRC-2014 분류 1위(top-5 test error 약 6.7%)를 달성했다.
AlexNet보다 파라미터가 약 12배 적은데(약 6~7M) 더 정확하다.

## 2. 핵심 아이디어

1. **Inception 모듈**: 한 층에서 1×1, 3×3, 5×5 conv와 3×3 max-pool을 **병렬**로 적용하고 채널 방향으로 concat. 여러 스케일의 패턴을 동시에 포착하고, 다음 층이 필요한 스케일을 선택하게 한다.
2. **1×1 conv를 통한 차원 축소(bottleneck)**: 3×3, 5×5 앞에 1×1 conv로 채널 수를 줄여 연산량을 크게 절감. (Network-in-Network의 1×1 conv 활용)
3. **Global Average Pooling**: 마지막 FC 대신 7×7 평균 풀링을 사용해 파라미터를 크게 줄이고 과적합을 억제(이후 분류기 FC는 1개만 유지).
4. **보조 분류기(Auxiliary classifiers)**: 중간층(4a, 4d 뒤)에 작은 분류기를 달아 학습 시 총 손실에 가중치 0.3으로 더한다. 깊은 망의 그래디언트 소실을 완화하고 정규화 효과를 기대. 추론 시에는 제거.
5. **연산 예산(1.5 billion multiply-adds)을 목표로 설계**: 실제 서비스 환경에서도 사용할 수 있는 효율성을 중시.

### 1×1 병목의 효과 (예: Inception 3a의 5×5 가지)

입력 192 채널, 출력 32 채널, 28×28 위치일 때 5×5 conv 곱셈 수(대략):

| 방식 | 계산 | 곱셈 수 |
|---|---|---|
| 직접 5×5 | 28·28·192·32·25 | 약 1.2억 |
| 1×1(→16) 후 5×5 | 28·28·192·16 + 28·28·16·32·25 | 약 0.024억 + 0.1억 ≈ 약 0.12억 |

→ 약 10배 절감. (위는 대략적인 계산)

## 3. 아키텍처/방법

### 3.1 Inception 모듈 (naive → dimension reduction)

```
         입력 (H×W×C_in)
  ┌─────────┬──────────────┬──────────────┬───────────────┐
 1×1 conv  1×1 conv(reduce) 1×1 conv(reduce) 3×3 max-pool(s1)
 (n1)       (n3r)            (n5r)            |
  |         3×3 conv (n3)    5×5 conv (n5)    1×1 conv (pool_proj)
  └─────────┴──────────────┴──────────────┴───────────────┘
              channel concat → 출력 C = n1 + n3 + n5 + pool_proj
```

모든 conv 뒤에 ReLU. 각 가지는 padding으로 H×W를 유지한다.

### 3.2 전체 구조 (입력 224×224×3)

| Layer | 설명 | 출력 크기 |
|---|---|---|
| conv1 | 7×7 /2, 64 (+ReLU) → 3×3 maxpool /2 | 56×56×64 |
| conv2 | 1×1, 64 → 3×3, 192 → 3×3 maxpool /2 | 28×28×192 |
| inception 3a | 64 / 96→128 / 16→32 / pool→32 | 28×28×256 |
| inception 3b | 128 / 128→192 / 32→96 / pool→64 | 28×28×480 |
| maxpool | 3×3 /2 | 14×14×480 |
| inception 4a | 192 / 96→208 / 16→48 / pool→64 | 14×14×512 |
| inception 4b | 160 / 112→224 / 24→64 / pool→64 | 14×14×512 |
| inception 4c | 128 / 128→256 / 24→64 / pool→64 | 14×14×512 |
| inception 4d | 112 / 144→288 / 32→64 / pool→64 | 14×14×528 |
| inception 4e | 256 / 160→320 / 32→128 / pool→128 | 14×14×832 |
| maxpool | 3×3 /2 | 7×7×832 |
| inception 5a | 256 / 160→320 / 32→128 / pool→128 | 7×7×832 |
| inception 5b | 384 / 192→384 / 48→128 / pool→128 | 7×7×1024 |
| avgpool | 7×7, stride 1 | 1×1×1024 |
| dropout | 40% | 1024 |
| linear | 1000 + softmax | 1000 |

(셀의 "a / b→c / d→e / pool→f" = 1×1 필터 수 / 3×3 reduce→3×3 / 5×5 reduce→5×5 / pool projection)

파라미터가 있는 층 기준 깊이 22층(풀링 제외).

### 3.3 보조 분류기

4a, 4d 출력에 각각: 5×5 avgpool(stride 3) → 1×1 conv 128 → FC 1024 + ReLU → dropout 0.7 → FC 1000.

총 손실:

```
L = L_main + 0.3 · L_aux1 + 0.3 · L_aux2
```

### 3.4 학습·평가 설정

- 비동기 SGD(DistBelief), momentum 0.9, 8 epoch마다 learning rate 4% 감소.
- 다양한 크기·종횡비의 패치 샘플링(원본 면적의 8%~100%, 종횡비 3/4~4/3) 증강.
- 테스트 시 **7개 모델 앙상블 × 144 crops** (4개 스케일 × 3 정사각형 × 6 크롭 × 좌우 반전)로 평균.

## 4. 실험 결과 요약

- ILSVRC-2014 분류: 1위, top-5 test error 약 6.67%. (2012 우승 AlexNet 약 16%, 2013 우승 Clarifai 약 11.7%, 같은 해 VGG 약 7.3%)
- 모델 수/크롭 수를 늘릴수록 성능 향상 (1 모델 1 crop → 7 모델 144 crop으로 갈수록 오류율 감소).
- 검출(Detection) 챌린지에서도 R-CNN 방식에 Inception을 backbone으로 적용해 1위 (약 43.9% mAP).

## 5. 의의 (앞뒤 논문과의 연결)

- **← VGG**: 같은 해 대회에서 VGG는 "단순하지만 무거운" 길, GoogLeNet은 "복잡하지만 효율적인" 길을 택했다. 두 접근은 이후 ResNet 계열에서 합쳐진다.
- **Network-in-Network의 계승**: 1×1 conv 병목과 global average pooling이 표준 기법으로 자리 잡음 (ResNet bottleneck block, MobileNet pointwise conv 등).
- **→ BatchNorm (다음 주)**: 이후 Inception v2/v3는 BN, 필터 분해(5×5 → 3×3 두 개, n×1과 1×n 분해), label smoothing 등으로 개선되었으며 BN 논문 자체가 Inception 구조 위에서 실험되었다.
- **→ ResNet**: 보조 분류기로 완화하려던 깊은 망의 최적화 문제를 skip connection이 근본적으로 해결. 이후 Inception-ResNet으로 결합.
- **→ MobileNet**: 채널 방향/공간 방향 연산을 분리하는 아이디어의 뿌리를 보여 준다.

## 6. 구현 노트

`implementation/model.py`는 위 표대로 Inception 모듈 9개를 구현한다.
- 원 논문의 LRN 층은 생략(현대 구현과 동일하게 제거).
- 보조 분류기는 학습 모드에서만 출력(`aux_logits=True`일 때 `(main, aux1, aux2)`), 평가 모드에서는 main만 반환.
- 보조 분류기의 5×5 avgpool(stride 3) 대신 `AdaptiveAvgPool2d(4)`를 사용해 입력 해상도에 상관없이 동작하게 했다 (224 입력에서는 논문과 동일하게 4×4가 된다).
- 최종 7×7 avgpool 대신 `AdaptiveAvgPool2d(1)` 사용.
- 풀링은 `ceil_mode=True`로 논문의 크기(56, 28, 14, 7)를 맞춤.
- 이 구현의 총 파라미터는 약 13.4M인데, 보조 분류기 2개(추론 시 제거)가 약 6.4M을 차지하고 본체는 약 7M이다.
- 가중치 초기화는 Kaiming, conv 뒤 BatchNorm은 사용하지 않는다(논문과 동일). `train.py`는 CIFAR-10을 64×64로 리사이즈해 보조 손실 포함 학습 루프를 보여 준다.
