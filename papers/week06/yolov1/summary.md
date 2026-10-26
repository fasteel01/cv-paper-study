# Week 6-2. YOLOv1

**You Only Look Once: Unified, Real-Time Object Detection**
Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi — *CVPR*, 2016 (arXiv 1506.02640)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

당시의 주류 검출기(DPM, R-CNN 계열)는 검출을 **여러 단계의 파이프라인**으로 풀었다.

- **DPM**: 슬라이딩 윈도우로 이미지 전체를 훑으며 분류기를 반복 적용.
- **R-CNN 계열**: 영역 제안 → 각 제안의 특징 추출 → 분류 → 박스 후처리/재점수화. 각 단계를 따로 학습하고 최적화해야 해서 **느리고 학습이 복잡**하다. (Fast R-CNN, Faster R-CNN이 속도를 개선했지만 여전히 두 단계 구조.)

YOLO의 문제의식: **검출을 하나의 회귀(regression) 문제로 재정의**하면 어떨까? 이미지 픽셀에서 곧바로 박스 좌표와 클래스 확률을 하나의 네트워크로 예측하면,
(1) 매우 빠르고(실시간), (2) 파이프라인이 단순하며, (3) 이미지 **전체를 한 번에 보기 때문에** 맥락 정보를 활용할 수 있다.

## 2. 핵심 아이디어

1. **통합된(unified) 검출**: 단일 CNN이 한 번의 forward로 모든 박스와 클래스 확률을 동시에 예측한다. "You Only Look Once."
2. **격자(grid) 기반 예측**: 이미지를 $S \times S$ 격자로 나누고, 객체 **중심이 속한 셀**이 그 객체를 검출할 책임을 진다.
3. 각 셀은 $B$개의 박스(좌표 + confidence)와 $C$개의 **클래스 조건부 확률**(셀당 하나의 클래스 분포)을 예측한다.
4. **Confidence** = $\Pr(\text{Object}) \cdot \text{IoU}^{\text{truth}}_{\text{pred}}$ — 박스 안에 객체가 있을 확률과 예측 박스의 정확도를 함께 나타낸다.
5. **전역적 추론**: Fast R-CNN은 배경 영역 패치를 객체로 오탐하는 배경 오류가 많지만, YOLO는 이미지 전체의 맥락을 보므로 배경 오류가 적다.
6. **단일 sum-squared error 손실**로 좌표/confidence/클래스를 한꺼번에 최적화한다(가중치로 균형 조절).

## 3. 아키텍처/방법

### 3.1 출력 텐서

PASCAL VOC 기준 $S = 7$, $B = 2$, $C = 20$ 이므로 출력은

$$S \times S \times (B \cdot 5 + C) = 7 \times 7 \times 30$$

- 박스당 5개 값: $(x, y, w, h, \text{confidence})$
  - $(x, y)$: 박스 중심, **해당 셀 경계 기준 상대 좌표** (0~1)
  - $(w, h)$: **이미지 전체 대비 비율** (0~1)
- 클래스 확률 $\Pr(\text{Class}_i \mid \text{Object})$ 는 **셀당 하나**(박스가 $B$개여도 공유).
- 테스트 시 클래스별 점수:

$$\Pr(\text{Class}_i \mid \text{Object}) \cdot \Pr(\text{Object}) \cdot \text{IoU}^{\text{truth}}_{\text{pred}} = \Pr(\text{Class}_i) \cdot \text{IoU}^{\text{truth}}_{\text{pred}}$$

### 3.2 네트워크 구조

GoogLeNet에서 영감을 받았으나 inception 모듈 대신 **1×1 reduction + 3×3 conv** 조합을 쓴다. **24개 conv + 2개 FC** (Fast YOLO는 9개 conv, 더 적은 필터).

| 블록 | 구성 | 출력 크기 |
|---|---|---|
| 입력 | 448×448×3 | |
| Conv1 + MaxPool | 7×7×64, stride 2 → 2×2 pool, s2 | 112×112×64 |
| Conv2 + MaxPool | 3×3×192 → 2×2 pool, s2 | 56×56×192 |
| Conv3–6 + MaxPool | 1×1×128, 3×3×256, 1×1×256, 3×3×512 → pool | 28×28×512 |
| Conv7–16 + MaxPool | [1×1×256, 3×3×512] ×4, 1×1×512, 3×3×1024 → pool | 14×14×1024 |
| Conv17–20 | [1×1×512, 3×3×1024] ×2 | 14×14×1024 |
| Conv21–24 | 3×3×1024, 3×3×1024 s2, 3×3×1024, 3×3×1024 | 7×7×1024 |
| FC1 | 4096 + Dropout 0.5 | 4096 |
| FC2 | 7×7×30 (선형) | 7×7×30 |

- 활성화: 마지막 층은 선형, 나머지는 **leaky ReLU** ($\phi(x) = x$ if $x>0$, 아니면 $0.1x$).
- **사전학습**: 앞 20개 conv + average pool + FC를 ImageNet 분류(224×224)로 약 1주 사전학습(top-5 약 88%). 이후 **conv 4개 + FC 2개를 추가**하고 입력 해상도를 **448×448**로 높여 검출용으로 미세조정한다(세밀한 시각 정보 필요).
- 박스 좌표는 이미지 크기로 정규화해 [0, 1]에 들어가게 한다.

### 3.3 손실 함수

모두 **sum-squared error**(최적화가 쉬운 형태)이고, 문제점들을 가중치로 보정한다.

$$
\begin{aligned}
L ={}& \lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}_{ij}^{obj}\big[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2\big] \\
&+ \lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}_{ij}^{obj}\big[(\sqrt{w_i}-\sqrt{\hat w_i})^2+(\sqrt{h_i}-\sqrt{\hat h_i})^2\big] \\
&+ \sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}_{ij}^{obj}(C_i-\hat C_i)^2 \\
&+ \lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}_{ij}^{noobj}(C_i-\hat C_i)^2 \\
&+ \sum_{i=0}^{S^2}\mathbb{1}_{i}^{obj}\sum_{c\in classes}(p_i(c)-\hat p_i(c))^2
\end{aligned}
$$

- $\mathbb{1}_{i}^{obj}$: 셀 $i$에 객체가 존재, $\mathbb{1}_{ij}^{obj}$: 셀 $i$의 $j$번째 박스가 그 예측을 **담당(responsible)**. 담당 박스는 셀 안의 $B$개 중 **GT와 IoU가 가장 높은** 박스이다 → 박스 예측기들이 특정 크기/비율에 특화되는 효과.
- $\lambda_{coord} = 5$: 좌표 오차의 비중을 키운다(분류 오차와 같은 비중이면 불균형).
- $\lambda_{noobj} = 0.5$: 객체가 없는 셀이 훨씬 많아 confidence가 0 쪽으로 쏠려 gradient를 압도하는 것을 막는다.
- **$\sqrt{w}, \sqrt{h}$**: 큰 박스의 작은 오차와 작은 박스의 작은 오차를 동일하게 취급하지 않도록(작은 박스에 더 민감하게) 하는 부분적 해결책. (IoU 손실이 이후 더 직접적인 해결책이 된다.)
- confidence의 타깃: 객체가 있는 담당 박스는 (예측 박스, GT)의 IoU, 없는 박스는 0.
- 클래스 오차는 **객체가 있는 셀에서만** 계산.

### 3.4 학습 설정

| 항목 | 값 |
|---|---|
| 데이터 | PASCAL VOC 2007 + 2012 (trainval) |
| epoch | 약 135 |
| 배치 크기 | 64 |
| optimizer | SGD, momentum 0.9, weight decay 0.0005 |
| learning rate | 처음 epoch에 $10^{-3} \to 10^{-2}$로 서서히 증가(warm-up; 처음부터 높으면 gradient 불안정으로 발산), 이후 $10^{-2}$ 75 epoch → $10^{-3}$ 30 epoch → $10^{-4}$ 30 epoch |
| 규제 | Dropout 0.5(FC1 뒤), 랜덤 스케일/이동(최대 20%), HSV 색공간 노출/채도 조정(최대 1.5배) |

### 3.5 추론과 한계

- 추론은 forward 한 번 + (필요시) NMS. 격자 구조가 공간 다양성을 강제하므로 NMS의 이득은 R-CNN/DPM보다 작다(mAP 약 2~3%p 향상).
- **공간적 제약**: 각 셀이 박스 $B$개와 **하나의 클래스**만 예측 → 한 셀에 작은 객체가 여럿 모여 있는 경우(새 떼 등) 검출이 어렵다. 
- **일반화 한계**: 박스를 데이터로부터 직접 학습하므로 훈련에 없던 종횡비/배치에서 일반화가 약하다. 또 conv를 여러 번 다운샘플한 **거친 특징**으로 박스를 예측한다.
- **손실의 한계**: 큰 박스와 작은 박스의 오차를 동일 취급하는 문제(√ 로 일부 완화) → 주요 오류는 **부정확한 localization**.

## 4. 실험 결과 요약

(수치는 논문 보고 값에 대한 개략적 서술이며, 정확한 값은 원문 표를 참고할 것.)

**PASCAL VOC 2007 — 실시간 검출기 비교**

| 모델 | 학습 데이터 | mAP | FPS |
|---|---|---|---|
| Fast YOLO | 07+12 | 약 52.7% | 약 155 |
| **YOLO** | 07+12 | 약 63.4% | 약 45 |
| DPM v5 (실시간 변형 등) | 2007 | 약 33% 안팎 | 최대 약 100 |
| Fast R-CNN | 07+12 | 약 70.0% | 약 0.5 |
| Faster R-CNN (VGG-16) | 07+12 | 약 73.2% | 약 7 |
| Faster R-CNN (ZF) | 07+12 | 약 62.1% | 약 18 |

- 실시간(≥30fps) 검출기 중에서 가장 높은 정확도. 지연 시간은 비디오 스트리밍 기준 25ms 미만.
- **오류 분석 (YOLO vs Fast R-CNN)**: YOLO는 localization 오류가 크고(전체 오류 중 약 19% 대 약 8.6%) **배경 오탐은 훨씬 적다**(약 4.75% 대 약 13.6%).
- **Fast R-CNN과의 앙상블**: 둘의 오류 유형이 보완적이어서 Fast R-CNN의 배경 오탐을 YOLO로 걸러내면 mAP가 약 71.8% → 약 75.0% 수준으로 향상된다(앙상블 속도 이득은 없음).
- **VOC 2012 test**: 약 57.9% mAP — 당시 R-CNN 계열보다 낮았고 특히 작은 객체에서 약했다.
- **일반화 (Picasso, People-Art 데이터셋)**: 자연 이미지로 학습해 그림·예술 작품으로 일반화할 때 R-CNN/DPM보다 성능 저하가 덜했다. → 객체의 형태·크기·관계 등 전역 정보를 학습한다는 근거.

## 5. 의의 (앞뒤 논문과의 연결)

- **이전: R-CNN → Fast → Faster R-CNN**: 이들은 "제안 + 분류"의 two-stage. YOLO는 **제안 단계 자체를 없애고** 격자 회귀로 직접 검출하는 **one-stage 검출기**의 출발점이다. 정확도는 낮지만 속도로 실시간 검출의 가능성을 열었다.
- **다음: SSD (Week 7)**: YOLO의 속도 아이디어에 **다중 스케일 특징 맵 + 앵커(default box)** 를 결합해 정확도를 크게 높인다.
- **RetinaNet (Week 8)**: one-stage가 two-stage보다 정확도가 낮은 이유가 극심한 클래스 불균형(배경 압도)임을 분석해 Focal Loss로 해결 — YOLO의 $\lambda_{noobj}$ 같은 휴리스틱을 원리적으로 대체하는 접근이다.
- **YOLO 시리즈**: YOLOv2(앵커·BatchNorm·고해상도 분류 사전학습), v3(다중 스케일), 이후 계열로 실시간 검출의 사실상 표준이 되었다.
- **한계의 교훈**: 격자 셀당 클래스 1개, 거친 특징, √ 근사 손실 → 이후 앵커, FPN, IoU 계열 손실 등으로 개선된다.

## 6. 구현 노트

`implementation/model.py`에 다음을 처음부터 구현했다.

- **`YOLOv1`**: 논문 Figure 3의 **24개 conv + 2개 FC**(leaky ReLU 0.1, Dropout 0.5, 마지막 선형), 입력 448×448 → 출력 `(N, 7, 7, 30)`. 파라미터 수는 약 2.7억(대부분 FC1의 50176×4096).
- **`encode_target`**: 정규화 `(cx, cy, w, h)` GT를 격자 타깃으로 변환($x, y$는 셀 내 상대 좌표, 셀당 객체 1개 제약).
- **`yolo_loss`**: 논문의 sum-squared 멀티파트 손실. 셀당 B개 예측 중 GT와 IoU가 최대인 것을 담당 predictor로 선택(벡터화), $\lambda_{coord}=5$, $\lambda_{noobj}=0.5$, $\sqrt{w}, \sqrt{h}$ 사용. confidence 타깃은 IoU(그래디언트 차단).
- **`decode_predictions`**: `confidence × 클래스 확률`로 클래스별 점수를 만들고 클래스별 NMS 수행.
- **현대적 단순화**: ImageNet 사전학습 단계를 생략하고 처음부터 구성(구조는 동일), 예측 폭이 음수일 때 √ 안정화를 위해 부호 유지 + epsilon 사용. 자세한 내용은 docstring 참고.
- `train.py`: 데이터셋이 무거우므로 **더미 이미지/GT**로 타깃 인코딩 → forward → 손실 → 3 step 학습 → 디코딩을 시연한다.
