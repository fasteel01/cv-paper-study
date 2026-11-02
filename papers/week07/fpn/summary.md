# Week 7-2. FPN

**Feature Pyramid Networks for Object Detection**
Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie — *CVPR*, 2017 (arXiv 1612.03144)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

객체 검출의 핵심 난제 중 하나는 **스케일 변화**다. 같은 클래스라도 이미지 속 크기가 수십 배 차이 난다. 당시의 대응책은 다음과 같았다.

| 방식 | 설명 | 한계 |
|---|---|---|
| (a) Featurized image pyramid | 이미지를 여러 크기로 리사이즈해 각각 CNN 통과 (DPM 시절 표준) | 추론 시간이 몇 배로 증가, 메모리 부담 큼 |
| (b) Single feature map | 마지막 conv 특징 하나만 사용 (Fast/Faster R-CNN, YOLOv1) | 강한 의미 정보는 있으나 해상도가 낮아 작은 객체에 약함 |
| (c) Pyramidal feature hierarchy | 백본 내부의 여러 층 특징을 각각 사용 (SSD) | 얕은 층은 해상도는 높지만 **의미(semantic) 정보가 약해** 검출 성능이 떨어짐 |

CNN은 이미 깊이에 따라 해상도가 줄어드는 **내재된 피라미드 구조**를 갖고 있지만, 층마다 의미 수준이 달라(semantic gap) 그대로 쓰기 어렵다.
FPN의 문제의식: *백본의 피라미드 구조를 이용하되, 모든 스케일에서 강한 의미를 가지는 특징 피라미드를 거의 추가 비용 없이 만들 수 없을까?*

## 2. 핵심 아이디어

1. **Top-down 경로 + Lateral 연결**: 가장 깊고 의미가 강한 특징을 위로(top-down) 2배씩 업샘플하며 내려보내고, 같은 해상도의 bottom-up 특징과 **원소별 합(element-wise add)** 으로 융합한다.
2. **모든 레벨이 같은 채널 수(256)**: 1×1 lateral conv로 채널을 통일하므로, 이후의 head(RPN, RoI head)가 **레벨 간 가중치를 공유**할 수 있다.
3. **3×3 smoothing conv**: 업샘플 + 합으로 생긴 aliasing 효과를 완화.
4. **레벨별 앵커 스케일 분담**: P2~P6 각 레벨에 **하나의 앵커 크기**(32, 64, 128, 256, 512)를 배정 — 스케일 담당을 레벨이 한다.
5. **RoI 레벨 할당 규칙**: 크기가 큰 RoI는 깊은 레벨, 작은 RoI는 얕은 레벨의 특징에서 추출한다.
6. 범용 부품이라 RPN, Fast R-CNN, 이후 Mask R-CNN, RetinaNet 등 거의 모든 검출/분할 모델에 붙는다.

## 3. 아키텍처/방법

### 3.1 Bottom-up / Top-down / Lateral

- **Bottom-up (백본)**: ResNet의 각 stage 마지막 출력을 $\{C_2, C_3, C_4, C_5\}$ 로 사용 (stride 4, 8, 16, 32). 고해상도의 $C_1$ 은 메모리 때문에 제외.
- **Top-down**: $M_5 = \text{conv}_{1\times1}(C_5)$ 에서 시작.

$$M_l = \text{conv}_{1\times1}(C_l) + \text{Upsample}_{\times 2}(M_{l+1}), \qquad P_l = \text{conv}_{3\times3}(M_l)$$

  업샘플은 nearest neighbor. 모든 $M_l, P_l$ 은 채널이 256이다 ($l = 2,3,4,5$).
- **$P_6$**: RPN의 매우 큰 앵커(512)를 위해 $P_5$ 에 stride 2의 subsampling(1×1 max pool)을 적용해 얻는다 (RoI head에서는 사용하지 않음).

| 레벨 | 출처 | stride | 앵커 크기 | 용도 |
|---|---|---|---|---|
| P2 | C2 + 업샘플 | 4 | 32² | RPN, RoI head |
| P3 | C3 + 업샘플 | 8 | 64² | RPN, RoI head |
| P4 | C4 + 업샘플 | 16 | 128² | RPN, RoI head |
| P5 | C5 | 32 | 256² | RPN, RoI head |
| P6 | P5 subsample | 64 | 512² | RPN 전용 |

### 3.2 FPN을 적용한 RPN

- 각 레벨에 동일한 RPN head(3×3 conv + objectness/box 1×1 conv)를 **공유**하여 적용. 앵커 종횡비는 $\{1:2, 1:1, 2:1\}$ 3개, 레벨당 크기 1개 → 총 15종 앵커.
- 라벨링 규칙은 Faster R-CNN과 동일(GT와 IoU ≥ 0.7이면 positive, < 0.3이면 negative, GT별 최고 IoU 앵커도 positive).
- 논문 실험: 레벨 간 head 가중치를 공유하든 말든 성능이 비슷 → 피라미드의 모든 레벨이 비슷한 의미 수준을 갖는다는 증거.

### 3.3 FPN을 적용한 Fast R-CNN: RoI 레벨 할당

폭 $w$, 높이 $h$ 인 RoI를 다음 레벨 $P_k$ 에서 추출한다.

$$k = \left\lfloor k_0 + \log_2\!\left(\frac{\sqrt{wh}}{224}\right) \right\rfloor, \qquad k_0 = 4$$

- ImageNet 사전학습의 표준 입력 크기 224를 기준으로, $224\times224$ 크기의 RoI는 $P_4$, $112\times112$ 는 $P_3$ 로 간다 ($k$는 [2, 5]로 제한).
- 레벨마다 RoIPool/RoIAlign으로 7×7을 뽑은 뒤, 공유되는 head(2개의 FC 1024)가 처리한다.

## 4. 실험 결과 요약

(논문 보고값 기준, 약 값. ResNet-50/101 백본, COCO)

- **RPN 평가 (COCO, Average Recall)**: 단일 스케일 $C_4$ RPN 대비, FPN을 쓰면 작은 객체 재현율 $AR_s$ 가 크게 상승 (약 +12.9 포인트 수준의 향상을 보고). 전체 AR@1000도 상승.
- **Fast R-CNN + FPN (ResNet-50)**: 단일 특징 맵 baseline 대비 AP가 약 +2~3 상승, 작은 객체(AP_s) 향상 폭이 가장 큼.
- **Faster R-CNN + FPN (ResNet-101, COCO test-dev)**: AP 약 36.2, AP@.5 약 59.1. 당시 단일 모델 기준 COCO 챌린지 우승팀들을 능가했고, 추가 기법 없이 달성.
- **Ablation 요지** (ResNet-50 RPN/Fast R-CNN 비교):
  - top-down 경로를 빼면(bottom-up 피라미드만) 성능 크게 하락 → **의미 정보 전달이 핵심**.
  - lateral 연결을 빼면(top-down만) 위치 정확도가 떨어짐 → 업샘플된 특징은 위치가 부정확하므로 **bottom-up의 정확한 위치 정보가 필요**.
  - 가장 해상도 높은 $P_2$ 하나만 사용하는 것보다 피라미드 전체를 사용하는 편이 좋음.
- 추가 연산 비용은 작아, 단일 스케일 baseline 대비 추론 시간 증가가 크지 않다 (image pyramid 방식보다 훨씬 빠름).

## 5. 의의 (앞뒤 논문과의 연결)

- **앞 (SSD)**: SSD는 백본의 여러 층 특징을 그대로 예측에 썼기 때문에 얕은 층(conv4_3)의 의미가 약했다(작은 객체 약점). FPN은 **top-down 경로로 고수준 의미를 얕은 층에 전달**하여 이 문제를 직접 해결한다. "Pyramidal feature hierarchy"(SSD) vs "Feature pyramid"(FPN)의 차이이다.
- **앞 (Faster R-CNN)**: 단일 스케일 특징에 의존하던 RPN/RoI head를 멀티스케일로 일반화한 것.
- **뒤 (RetinaNet)**: 다음 주 RetinaNet은 FPN($P_3$~$P_7$) 위에 dense 원스테이지 검출기를 얹고 Focal Loss로 클래스 불균형을 해결한다.
- **뒤 (Mask R-CNN)**: Mask R-CNN의 표준 구현도 ResNet-FPN을 백본으로 쓴다. 이후 PANet, BiFPN, 그리고 Swin/ViT 계열 검출기(Mask R-CNN + Swin 등)에서도 FPN류 neck이 표준 부품으로 남는다.

## 6. 구현 노트

`implementation/model.py` 에 다음을 구현했다.
- `ResNet50Backbone`: bottleneck 블록을 직접 쌓아 $C_2$~$C_5$ 반환
- `FPN`: lateral 1×1, top-down `F.interpolate(nearest)` + add, 3×3 smoothing, $P_6$ = stride-2 max-pool
- `generate_anchors`, `RPNHead`(레벨 간 가중치 공유), `propose`(레벨별 top-k → 디코딩 → NMS)
- `map_rois_to_levels`: 위 식의 $k$ 계산, `MultiLevelRoIAlign`: 레벨별로 RoI를 모아 추출, `BoxHead`: FC 1024 × 2 + cls/reg

원 논문과의 차이(자세한 내용은 model.py docstring): RoIPool 대신 `torchvision.ops.roi_align`, NMS는 `torchvision.ops.nms`, 사전학습 가중치 없음, frozen BN 대신 일반 BN.
`train.py` 는 더미 데이터로 RPN 손실(앵커 라벨링/샘플링) + box head 손실을 계산해 학습 step을 보여준다.
