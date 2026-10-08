# Week 3-1. Batch Normalization

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**
Sergey Ioffe, Christian Szegedy — *ICML*, 2015
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

- 깊은 망을 SGD(미니배치)로 학습할 때, 앞쪽 레이어의 파라미터가 바뀌면 뒤쪽 레이어에 들어오는 입력의 분포가 계속 변한다.
  저자들은 이를 **Internal Covariate Shift**라 부른다. 뒤쪽 레이어는 "움직이는 목표"에 계속 적응해야 한다.
- 입력 분포가 변하면 sigmoid/tanh 같은 포화(saturating) 비선형성에서 그래디언트 소실이 심해진다.
  그래서 당시에는 **낮은 학습률, 신중한 가중치 초기화, ReLU 사용**이 필수였고 학습이 느렸다.
- 입력 이미지를 whitening(평균 0, 분산 1 등)하면 학습이 빨라진다는 것은 알려져 있었다.
  이 아이디어를 **망 내부의 각 레이어 입력에도** 적용하자는 것이 출발점이다.
- 단순히 활성값을 정규화한 뒤 그래디언트 계산에서 그 정규화를 무시하면, 파라미터 업데이트가 정규화 효과를 상쇄하며 값이 발산할 수 있다.
  따라서 정규화를 **미분 가능한 연산으로 망 안에 포함**시켜 역전파가 그 과정을 반영하도록 해야 한다.

## 2. 핵심 아이디어

1. **미니배치 통계로 정규화**: 각 활성값(채널)을 현재 미니배치의 평균·분산으로 정규화한다. 전체 데이터의 whitening은 비용이 크기 때문에, 미니배치 통계를 근사로 쓴다.
2. **학습 가능한 scale(γ)과 shift(β)**: 정규화만 하면 표현력이 줄어들 수 있다(예: sigmoid의 선형 영역에 갇힘).
   그래서 `γ·x̂ + β`로 되돌릴 수 있는 자유도를 준다. γ=√Var, β=E[x]이면 항등 변환도 복원 가능하다.
3. **추론 시에는 고정된 통계 사용**: 학습 중 이동평균(running mean/var)을 모아 두었다가, 추론 때는 배치에 의존하지 않는 결정적 연산으로 쓴다.
4. 부수 효과: 높은 학습률 사용 가능, 초기화에 덜 민감, 약한 **정규화(regularization) 효과**(배치 구성에 따른 노이즈)로 Dropout 필요성 감소.

## 3. 방법

### 3.1 학습 시 변환 (미니배치 B = {x₁ … x_m}, 한 활성값 기준)

| 단계 | 수식 |
|---|---|
| 배치 평균 | μ_B = (1/m) Σᵢ xᵢ |
| 배치 분산 | σ²_B = (1/m) Σᵢ (xᵢ − μ_B)² |
| 정규화 | x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε) |
| scale & shift | yᵢ = γ·x̂ᵢ + β |

- 완전연결층에서는 뉴런마다, **합성곱층에서는 채널마다** 하나의 (γ, β)를 두고, 통계는 배치 × 공간(H×W) 전체에서 계산한다
  (합성곱의 성질을 보존하기 위함). 채널 C개이면 학습 파라미터는 2C개.
- 역전파는 μ_B, σ²_B가 xᵢ에 의존한다는 점까지 체인룰로 반영한다 (PyTorch autograd가 자동 처리).

### 3.2 추론 시 변환

- 학습 중 `running_mean ← (1−momentum)·running_mean + momentum·μ_B`, `running_var`도 동일하게 갱신.
- 추론: `y = γ·(x − running_mean)/√(running_var + ε) + β` — 입력에 대한 **선형 변환**이므로 직전 Conv/FC에 **융합(fuse)** 가능해 추가 비용이 거의 없다.
- (논문은 추론용 분산에 불편 분산 m/(m−1) 보정을 쓴다고 서술한다. 구현체마다 세부는 다름.)

### 3.3 배치 위치

- 논문은 비선형성 **앞**, 즉 `Conv → BN → ReLU` 순서를 제안한다. 이때 Conv의 bias는 BN의 β가 대신하므로 불필요하다(`bias=False`).

### 3.4 ImageNet 적용 시 함께 쓴 변경 (Inception 기반)

- 학습률을 크게 올림(논문에서 5배, 30배 등 실험), Dropout 제거, L2 가중치 감쇠 축소,
  학습률 감쇠 가속, LRN 제거, 학습 샘플을 더 철저히 섞기(shuffle), 광학적 왜곡 감소.

## 4. 실험 결과 요약

- **MNIST 소규모 실험**: BN을 쓴 망이 더 빨리, 더 높은 정확도로 수렴하고, sigmoid 입력 분포가 학습 내내 안정적임을 보임.
- **ImageNet (Inception 변형)**: BN을 적용하면 기본 Inception과 같은 정확도에 **약 14배 적은 학습 스텝**으로 도달.
  학습률을 높인 BN-x5, BN-x30 변형은 더 높은 정확도에 도달했고, BN-x30은 sigmoid 비선형성으로도 학습 가능함을 보임.
- **앙상블**: BN-Inception 앙상블로 ImageNet top-5 오류율 약 4.9%(당시 인간 평가자 수준을 넘어선 것으로 보고)를 달성.
- (참고: 이후 연구들은 BN의 이득이 "internal covariate shift 감소"보다는 **손실 지형의 평활화(smoothing)** 때문이라고 주장한다. Santurkar et al., 2018.)

## 5. 의의

- 앞선 논문과의 연결: AlexNet의 LRN을 대체하고, VGG처럼 초기화·단계적 학습이 까다롭던 깊은 망, GoogLeNet의 보조 분류기 같은 "깊이를 위한 트릭"의 필요를 줄였다.
- 뒤 논문과의 연결: **ResNet**(다음 논문)은 모든 Conv 뒤에 BN을 사용하여 100층 이상의 망을 안정적으로 학습했고, 이후 거의 모든 CNN의 기본 부품이 되었다.
- 한계: 배치 크기가 작거나(검출·분할의 큰 해상도) 순환 구조에서는 통계가 불안정 → LayerNorm, GroupNorm, InstanceNorm 등으로 이어지며(Transformer는 LayerNorm),
  학습/추론 동작이 다르다는 점(`model.train()` / `model.eval()`)이 흔한 버그의 원인이다.

## 6. 구현 노트

- `implementation/model.py`: `nn.BatchNorm2d`를 쓰지 않고 **BatchNorm2d를 직접 구현**(미니배치 통계 + running 통계 + γ/β)했다.
  `torch.nn.BatchNorm2d`와 출력이 일치하는지 `__main__`에서 검증한다.
- 비교용 작은 CNN(`SmallCNN`)은 `use_bn` 플래그로 BN 유무를 전환한다.
- `implementation/train.py`: CIFAR-10에서 BN 유무 및 학습률별 학습 곡선을 비교한다.
- 주의: 평가 시 `model.eval()`을 꼭 호출, 학습 시 배치 크기가 너무 작지 않게 할 것.
