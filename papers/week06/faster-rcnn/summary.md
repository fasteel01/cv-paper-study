# Week 6-1. Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun — *NeurIPS*, 2015 (arXiv 1506.01497)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

Fast R-CNN은 conv 특징을 이미지당 한 번만 계산해 검출 네트워크 자체를 매우 빠르게 만들었지만, 영역 제안(region proposal)은 여전히 **CPU에서 도는 Selective Search**(이미지당 약 2초)에 의존했다.
그 결과 제안 생성이 전체 시간의 병목이 되었고, 시스템은 end-to-end가 아니었다.

- 제안 생성기(Selective Search, EdgeBoxes 등)는 검출 네트워크와 별개의 알고리즘이어서 **특징을 공유하지 못한다**.
- GPU로 옮겨 다시 구현하면 빨라질 수 있지만, 하류 검출기와 계산을 공유하는 기회를 놓친다.

Faster R-CNN의 질문: **"제안 생성도 conv 특징 위에서 학습되는 작은 네트워크로 만들 수 있지 않을까?"**
conv 특징 맵은 이미 객체의 위치 정보를 담고 있으므로, 그 위에 몇 개의 conv 층만 얹으면 제안을 거의 공짜로(이미지당 약 10ms) 얻을 수 있다는 것이 핵심 관찰이다.

## 2. 핵심 아이디어

1. **Region Proposal Network (RPN)**: conv 특징 맵 위를 3×3 창으로 슬라이딩하며 각 위치에서 "객체인가 아닌가(objectness)"와 "박스 보정값"을 예측하는 fully-convolutional 네트워크.
   검출 네트워크(Fast R-CNN)와 **conv 층을 공유**하므로 제안 비용이 사실상 사라진다.
2. **앵커(Anchor) 박스**: 각 슬라이딩 위치에 미리 정의한 여러 크기·비율의 기준 박스 $k$개(논문: 3 스케일 × 3 비율 = 9개)를 두고, 이 기준 대비 보정값을 회귀한다.
   이미지/필터 피라미드 없이 **단일 스케일 특징 맵으로 다중 스케일·비율**을 다루는 효율적인 방법이다.
3. **번역 불변성(translation invariance)**: 앵커와 이를 계산하는 함수가 모두 위치에 무관하게 같은 가중치를 공유한다. MultiBox 방식(800×(4+1) 출력의 FC층) 대비 파라미터가 훨씬 적다.
4. **RPN과 Fast R-CNN의 특징 공유 학습**: 두 네트워크가 conv 층을 공유하도록 학습한다(논문은 4단계 교대 학습 제안, 이후 approximate joint 학습이 널리 쓰임).

## 3. 아키텍처/방법

### 3.1 전체 구조

```
입력 이미지 ─► 공유 conv layers ─► conv feature map ──┬─► RPN ─► 제안 박스(~300개, NMS 후)
                                                      │                    │
                                                      └─► RoI Pooling ◄────┘
                                                              ▼
                                                  FC6 ─► FC7 ─► (K+1) cls / 4K reg
```

하나의 통합 네트워크이며, RPN이 Fast R-CNN 헤드에게 "어디를 볼지" 알려주는 attention 역할을 한다.

### 3.2 RPN 구조

- 입력: conv 특징 맵(VGG-16이면 conv5_3, stride 16, 512채널; ZF는 256채널).
- 3×3 conv(중간층 512-d 또는 256-d) + ReLU → 두 개의 형제 1×1 conv:
  - **cls 층**: 앵커당 2개 점수(객체/배경) → $2k$ 채널
  - **reg 층**: 앵커당 4개 보정값 → $4k$ 채널
- 3×3 창의 수용 영역(receptive field)이 크기 때문에(VGG-16에서 약 228 픽셀) 입력 이미지의 큰 영역 정보를 본다.
- 특징 맵 크기가 약 $W \times H \approx 2400$(600×1000 입력, stride 16)이면 총 앵커 수는 약 $W H k \approx 20000$.

### 3.3 앵커

| 항목 | 값 |
|---|---|
| 스케일 (면적 기준 한 변 길이) | $128^2,\ 256^2,\ 512^2$ 픽셀 |
| 종횡비 | 1:1, 1:2, 2:1 |
| 앵커 수 $k$ | 9 |
| 앵커 중심 | 슬라이딩 창 중심(입력 이미지 좌표로 매핑) |

### 3.4 앵커 라벨링과 손실

**양성(positive)** 라벨:
1. 어떤 GT 박스와 **IoU가 가장 높은** 앵커(들) — 모든 GT가 적어도 하나의 양성 앵커를 갖도록 보장
2. 어떤 GT 박스와 **IoU > 0.7** 인 앵커

**음성(negative)**: 모든 GT와 IoU < 0.3 인 비양성 앵커. 그 사이(0.3~0.7)이거나 양/음성이 아닌 앵커는 학습에 기여하지 않는다.
**이미지 경계를 벗어나는 앵커는 학습 시 무시**한다(그렇지 않으면 수렴하지 않는 큰 오차 항이 생김). 테스트 시에는 경계 밖 제안을 이미지 안으로 클리핑한다.

Multi-task loss (이미지당):

$$L(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda\,\frac{1}{N_{reg}}\sum_i p_i^*\, L_{reg}(t_i, t_i^*)$$

- $p_i$: 앵커 $i$가 객체일 예측 확률, $p_i^*$: 정답 라벨(양성 1, 음성 0)
- $L_{cls}$: 두 클래스 log loss, $L_{reg}(t, t^*) = \text{smooth}_{L_1}(t - t^*)$ (양성 앵커에만 적용, $p_i^*$ 곱)
- 미니배치 크기 $N_{cls} = 256$ (이미지 1장에서 샘플링), 양성:음성 = 최대 1:1 (양성이 모자라면 음성으로 채움), $N_{reg} \approx 2400$ (앵커 위치 수), $\lambda = 10$ → 두 항이 대략 같은 가중치. 논문은 결과가 $\lambda$에 민감하지 않다고 보고한다.

**박스 파라미터화** (R-CNN과 동일 계열, 앵커 $(x_a, y_a, w_a, h_a)$ 기준):

$$t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}$$

(GT $t^*$도 동일 식으로 계산.) 각 앵커 모양(k개)마다 **독립적인 회귀기**가 있어 서로 다른 크기·비율의 박스를 각자 담당한다(가중치 공유 안 함).

### 3.5 제안 생성(Proposal layer)과 NMS

1. 모든 앵커에 회귀값을 적용해 제안 박스를 만들고 이미지 경계로 클리핑
2. 객체 점수(cls score) 기준으로 정렬
3. **NMS (IoU 임계값 0.7)** 로 중복 제거 — 약 6000개 → 약 2000개 수준으로 줄이고, 이후 상위 N개(테스트 시 300개)를 검출 헤드에 넘긴다.

NMS는 mAP에 거의 영향을 주지 않으면서 제안 수를 대폭 줄인다.

### 3.6 RPN과 Fast R-CNN의 특징 공유 학습

| 방식 | 내용 |
|---|---|
| **4단계 교대 학습** (논문 채택) | ① ImageNet 사전학습 모델로 RPN 학습 ② ①의 제안으로 별도의 Fast R-CNN 학습(아직 conv 미공유) ③ 검출기 conv를 가져와 RPN 초기화, 공유 conv는 고정하고 RPN 고유 층만 미세조정 ④ 공유 conv 고정, Fast R-CNN 고유 층만 미세조정 |
| Approximate joint training | 하나의 네트워크로 RPN 손실 + Fast R-CNN 손실 합을 한 번에 역전파. 제안 박스 좌표에 대한 미분은 무시. 학습 시간을 약 25~50% 줄이면서 비슷한 정확도(후속 구현에서 표준이 됨) |
| Non-approximate joint training | 박스 좌표에 대한 미분까지 포함(RoI warping 층 필요) |

### 3.7 Fast R-CNN 헤드

RPN 제안을 RoI Pooling(7×7)으로 고정 크기 특징으로 만들고 FC6/FC7 → (K+1) 분류 + 클래스별 박스 회귀를 수행한다(Fast R-CNN과 동일).
학습 시 미니배치는 이미지 2장(논문의 4단계 학습 기준)에서 RoI 128개, 양성 25%.

## 4. 실험 결과 요약

(수치는 논문 보고 값에 대한 개략적 서술이며, 정확한 값은 원문 표를 참고할 것.)

| 설정 | 제안 생성 | VOC07 test mAP | 속도 |
|---|---|---|---|
| Fast R-CNN (VGG-16) | Selective Search (2000개) | 약 66.9% | 제안 포함 이미지당 약 2초 이상 |
| **Faster R-CNN (ZF)** | RPN (300개), 공유 | 약 59.9% | 약 17 fps |
| **Faster R-CNN (VGG-16)** | RPN (300개), 공유 | 약 69.9% (VOC07 학습) / 약 73.2% (VOC07+12 학습) | 약 5 fps (이미지당 약 198ms 수준) |

- 공유 없이 RPN만 쓰는 것도 Selective Search 대비 동등 이상이며, 제안 수를 줄여도(300개) mAP가 거의 유지된다 → 제안 품질이 높다.
- **속도**: 공유 conv 덕분에 제안 비용이 이미지당 약 10ms 수준으로 줄어, 제안 포함 전체가 VGG-16 기준 약 5fps(Selective Search 기반은 약 0.5fps)로 빨라졌다.
- **앵커 분석 (ablation)**: 단일 스케일/비율 앵커보다 3×3 조합이 더 좋고, 스케일 다양성이 비율 다양성보다 더 중요하다.
- **IoU 제안 품질(recall)**: 제안 수가 줄어도 RPN은 Selective Search/EdgeBoxes보다 recall 하락이 완만하다.
- MS COCO에서도 VGG-16으로 AP@0.5 약 42%, AP@[.5:.95] 약 21~22% 수준을 보고하며, 이후 ILSVRC/COCO 2015 대회 우승 모델(ResNet-101 기반)의 기반이 되었다.
- 후속 연구에서 ResNet-101 백본을 쓴 Faster R-CNN이 COCO에서 큰 폭의 성능 향상을 보였다.

## 5. 의의 (앞뒤 논문과의 연결)

- **Fast R-CNN → Faster R-CNN**: 마지막 외부 모듈(Selective Search)을 신경망(RPN)으로 대체하며 **거의 end-to-end 학습 가능한 two-stage 검출기**가 완성되었다.
- **앵커 개념의 확산**: 앵커 기반 예측은 이후 SSD, RetinaNet, YOLOv2/v3 등 단일 단계 검출기의 핵심 장치가 되었다. (다음 논문 **YOLOv1**은 앵커 없이 격자 기반 회귀로 "제안 단계 자체를 없애는" 반대 접근이다.)
- **FPN (Week 7)**: 단일 스케일 특징 맵의 한계를 다중 스케일 피라미드로 보완하고, RPN을 각 피라미드 레벨에 적용한다.
- **Mask R-CNN (Week 9)**: Faster R-CNN에 RoIAlign과 마스크 브랜치를 얹은 구조다.
- **한계**: 두 단계(제안 → 분류)라 단일 단계 검출기보다 느리고, NMS 등 수작업 후처리와 앵커 하이퍼파라미터(스케일·비율)에 의존한다. 이 의존성은 훗날 DETR이 제거하려 한다.

## 6. 구현 노트

`implementation/model.py`에 다음을 처음부터 구현했다.

- **`generate_anchors`**: 특징 맵의 각 위치에 3 스케일 × 3 비율 = 9개의 앵커를 (y, x, anchor) 순서로 생성.
- **`RPN`**: 3×3 conv + 1×1 cls(2k) / reg(4k). 출력을 `(N, H*W*k, ·)`로 reshape하여 앵커 순서와 일치시킨다.
- **`assign_anchor_targets` / `rpn_loss`**: IoU 0.7/0.3 기준 양성·음성 라벨링, 경계 밖 앵커 무시, 이미지당 256개 샘플링(양성 ≤ 128), CE + smooth L1.
- **`generate_proposals`**: 디코딩 → 클리핑 → 작은 박스 제거 → 직접 구현한 **NMS(0.7)**.
- **검출 헤드**: RoIPool(7×7, 순수 PyTorch) + FC + 분류/회귀 형제 층, `sample_rois`(GT를 후보에 추가), 클래스별 NMS를 하는 추론 경로.
- **현대적 단순화**: ImageNet 사전학습 VGG-16 대신 작은 conv 백본(stride 16), 4단계 교대 학습 대신 **approximate joint training**(제안을 detach), 단일 크기 이미지 배치. 자세한 내용은 docstring 참고.
- `train.py`: 더미 이미지·GT로 RPN 손실 + 검출 손실을 합산하여 1~3 step 학습하는 데모.
