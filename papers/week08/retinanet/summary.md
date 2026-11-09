# Week 8-1. RetinaNet (Focal Loss)

**Focal Loss for Dense Object Detection**
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár — *ICCV*, 2017 (arXiv 1708.02002)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

당시 검출기는 정확도와 속도의 트레이드오프가 뚜렷했다.

| 계열 | 대표 | 특징 |
|---|---|---|
| Two-stage | R-CNN 계열, Faster R-CNN, FPN | 1단계(RPN 등)에서 후보를 약 1~2천 개로 줄인 뒤 2단계에서 분류. 정확하지만 느림 |
| One-stage | YOLOv1, SSD | 이미지 전체를 촘촘한 앵커(~10⁴~10⁵개)로 한 번에 분류·회귀. 빠르지만 정확도가 two-stage에 못 미침 |

논문의 핵심 진단: **one-stage 검출기가 뒤처지는 주된 이유는 학습 중 극단적인 foreground–background 클래스 불균형**이다.

- 이미지당 앵커는 수만 개인데 객체에 매칭되는 앵커(positive)는 수십 개 이하 → 대다수가 **쉬운 negative(easy negative)**.
- Two-stage는 (1) 후보 영역 제안 단계에서 대부분의 background를 걸러내고, (2) 2단계에서 샘플링(positive:negative = 1:3)이나 OHEM으로 균형을 맞춘다.
- One-stage는 모든 앵커를 손실에 넣으므로, 쉬운 negative 하나하나의 손실은 작아도 **개수가 압도적이라 합이 손실과 gradient를 지배**하여 학습이 비효율적이고 모델이 퇴화(degenerate)한다.
- SSD는 이를 hard negative mining(3:1)으로 해결했지만, 이는 샘플을 버리는 방식이다. RetinaNet은 **손실 함수 자체를 바꿔** 모든 앵커를 쓰되 쉬운 샘플의 기여를 자동으로 줄인다.

## 2. 핵심 아이디어

1. **Focal Loss**: 교차 엔트로피에 변조 항 $(1-p_t)^\gamma$ 를 곱해, 이미 잘 분류된 샘플(큰 $p_t$)의 손실을 기하급수적으로 줄이고 어려운 샘플에 학습을 집중시킨다.
2. **RetinaNet**: 구조는 의도적으로 단순한 one-stage 검출기 — **ResNet + FPN 백본, 두 개의 병렬 서브넷(분류/박스 회귀)**. 정확도 향상은 구조가 아니라 Focal Loss 덕분임을 보이는 것이 목적.
3. **Prior 초기화**: 분류 출력 층의 bias를 $b=-\log\frac{1-\pi}{\pi}$, $\pi=0.01$ 로 초기화해 학습 초기 모든 앵커의 foreground 확률을 약 0.01로 만든다. 초기에 압도적 다수 negative에서 나오는 큰 손실로 인한 불안정을 방지.
4. **Sigmoid 기반 K-way 이진 분류**: softmax 대신 클래스마다 독립적인 sigmoid. 배경은 별도 클래스가 아니라 "모든 클래스 출력이 0".

## 3. 아키텍처/방법

### 3.1 Focal Loss

이진 CE는 $\text{CE}(p_t) = -\log(p_t)$ 이며, $p_t = p$ (y=1), $1-p$ (y=0).

$$\text{FL}(p_t) = -\alpha_t\,(1-p_t)^{\gamma}\,\log(p_t)$$

- $\gamma \ge 0$ : focusing 파라미터. $\gamma=0$ 이면 (α-balanced) CE.
- $p_t=0.9$ 인 쉬운 샘플은 $\gamma=2$ 에서 CE 대비 손실이 $(0.1)^2 = 1/100$ 로 감소, $p_t \approx 0.968$ 이면 약 1/1000. 반면 $p_t$ 가 작은 어려운 샘플은 거의 영향이 없다.
- $\alpha_t$ : 클래스 가중치 ($\alpha$ for positive, $1-\alpha$ for negative). 논문의 최적값은 **$\alpha=0.25$, $\gamma=2$** — γ가 커질수록 쉬운 negative가 줄어들어 α는 오히려 작아지는 것이 최적이라는 점이 흥미롭다(positive의 가중을 낮춤).
- 전체 손실은 **모든 앵커에 대한 합을 "매칭된(positive) 앵커 수"로 정규화** (전체 앵커 수가 아님: 대부분의 앵커는 손실이 거의 0이므로).

### 3.2 구조

| 구성요소 | 내용 |
|---|---|
| 백본 | ResNet-50/101 |
| 피라미드 | FPN $P_3\sim P_7$ (stride 8~128, 채널 256). $P_3\sim P_5$ 는 $C_3\sim C_5$ 에서 top-down + lateral, $P_6$ 은 $C_5$ 위의 3×3 stride-2 conv, $P_7$ 은 ReLU 후 3×3 stride-2 conv. FPN 논문과 달리 고해상도 $P_2$ 는 연산량 문제로 사용하지 않음 |
| 앵커 | 레벨별 면적 $32^2\sim512^2$ ($P_3\sim P_7$), 종횡비 {1:2, 1:1, 2:1} × 스케일 {$2^0, 2^{1/3}, 2^{2/3}$} = **9개/위치** |
| 분류 서브넷 | 3×3 conv(256)+ReLU 4번 → 3×3 conv($K\cdot A$) → sigmoid. 모든 레벨 간 가중치 공유 |
| 박스 서브넷 | 분류 서브넷과 같은 구조이되 **별개의 파라미터**, 출력 $4A$ (class-agnostic 박스 회귀) |
| 앵커 라벨 | GT와 IoU ≥ 0.5 → positive, [0, 0.4) → negative, [0.4, 0.5) → 무시 |
| 박스 회귀 | RPN과 같은 $(t_x,t_y,t_w,t_h)$ 파라미터화, smooth L1 손실 |
| 추론 | 레벨별로 점수 > 0.05 인 상위 1000개 예측만 디코딩 → 모든 레벨 병합 후 클래스별 NMS(0.5) |

서브넷이 FC가 아니라 conv이므로 입력 크기에 무관하고, 공간 위치마다 같은 판별 규칙을 적용한다 (RPN, SSD와 같은 dense 예측).

### 3.3 왜 Focal Loss가 필요한가 (수치 감각)

- 초기화 후 모든 앵커의 $p\approx0.01$ 이면, negative 앵커의 CE는 $-\log(0.99)\approx0.01$ 로 작지만 수만 개가 합쳐지면 positive 수십 개의 손실(≈ $-\log 0.01 \approx 4.6$)을 압도한다.
- FL($\gamma=2$)에서는 negative 하나의 손실이 $0.99$ 대신 $(0.01)^2$ 배, 즉 약 $10^{-4}$ 배로 줄어 합이 positive 항 수준으로 내려간다. 논문 분석에서도 학습된 모델에서 FL은 쉬운 negative의 누적 손실 비중을 극적으로 낮춘다 (CE는 거의 전부를 쉬운 negative가 차지).

## 4. 실험 결과 요약

(논문 보고값 기준, 약 값. COCO)

- **Ablation (ResNet-50, 600 px)**: CE → α-balanced CE로 약 0.9 AP 개선, $\gamma=2$ 의 FL로 추가 개선하여 약 +3 AP 수준. 최종 설정 α=0.25, γ=2.
- **OHEM, 샘플링 기반 기법과 비교**: 같은 아키텍처에서 FL이 OHEM 계열을 능가 (약 3 AP 이상 차이).
- **Prior 초기화**: 없으면 학습이 발산하거나 불안정.
- **최종 (test-dev)**: RetinaNet-101-800 이 AP 약 **39.1** — 당시 two-stage 대표 Faster R-CNN w/ FPN (약 36.2) 및 다른 one-stage 검출기(DSSD, YOLO류, SSD 계열)를 정확도에서 앞섬. 속도는 R-CNN 계열보다 빠른 영역에서 경쟁력 있는 속도-정확도 곡선을 형성.
- 해상도/백본을 줄이면 (예: ResNet-50, 500 px) 더 빠른 추론이 가능하며 같은 속도대에서 Faster R-CNN류보다 높은 AP.

## 5. 의의 (앞뒤 논문과의 연결)

- **앞 (SSD / YOLOv1)**: 이들의 정확도 한계 원인을 "구조"가 아니라 "클래스 불균형"으로 규명했다. SSD의 hard negative mining(3:1)을 연속적인 가중 함수로 대체한 셈.
- **앞 (FPN)**: 같은 저자 그룹의 FPN을 one-stage 검출기의 neck으로 그대로 채택($P_3\sim P_7$). FPN의 "멀티스케일 의미 특징" + Focal Loss의 조합이 one-stage의 승리 공식.
- **앞 (Faster R-CNN)**: RPN의 앵커를 더 촘촘히(9개/위치×5레벨) 쓰고 RPN-스타일 head에 K-way 분류를 직접 붙인 단일 단계 구조.
- **뒤 (FCN / 분할)**: Focal Loss는 클래스 불균형이 심한 dense prediction 일반(semantic segmentation, 의료영상 등)에 널리 이식된다.
- **뒤 (Mask R-CNN, DETR)**: Anchor-free 계열(FCOS, CenterNet 등)도 Focal Loss를 표준 분류 손실로 사용했고, DETR 계열 손실(예: Deformable DETR 등)에서도 focal 형태의 분류 손실이 쓰인다.

## 6. 구현 노트

`implementation/model.py` 에 다음을 구현했다.
- `ResNet50Backbone`: bottleneck을 직접 쌓아 $C_3\sim C_5$ 반환
- `FPNP3P7`: lateral 1×1 + top-down + 3×3 smoothing, $P_6$ = $C_5$ 에서 stride-2 conv, $P_7$ = ReLU 후 stride-2 conv
- `generate_anchors`: 레벨 × (3 종횡비 × 3 스케일) = 9 앵커, head의 flatten 순서와 일치
- `RetinaHead`: 4×conv+ReLU 서브넷 2개(분류/박스), 레벨 간 공유, 분류 bias를 $-\log((1-\pi)/\pi)$ 로 초기화
- `sigmoid_focal_loss`: $p_t$, $\alpha_t$ 를 이용한 직접 구현. `detect()` : 점수 임계 → top-k → 디코딩 → `batched_nms`

원 논문과의 차이(자세한 내용은 model.py docstring): 일반 BN(논문은 frozen BN), 사전학습 가중치 없음, NMS는 `torchvision.ops.batched_nms`.
`train.py` 는 더미 데이터로 앵커 매칭(0.5/0.4, 사이는 무시) → Focal Loss(positive 수로 정규화) + smooth L1 → 1 step 학습을 보이고,
마지막에 BCE와 Focal Loss에서 negative 항이 전체 손실에서 차지하는 비율을 비교해 출력한다.
