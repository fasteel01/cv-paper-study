# Week 5-2. Fast R-CNN

**Fast R-CNN**
Ross Girshick — *ICCV*, 2015 (arXiv 1504.08083)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

R-CNN은 정확도 면에서 탐지 성능을 크게 끌어올렸지만 실용성 측면에서 다음 문제가 있었다.

1. **다단계 학습 파이프라인**: CNN 미세조정(log loss) → 선형 SVM 학습 → 박스 회귀기 학습. 각 단계가 분리되어 있다.
2. **학습이 공간·시간 비용이 큼**: SVM과 회귀기 학습을 위해 각 제안 영역의 특징을 **디스크에 저장**해야 한다(VGG-16 기준 수백 GB).
3. **테스트가 느림**: 이미지당 약 2000개 제안을 각각 CNN에 통과시킨다(VGG-16 기준 이미지당 약 47초). 제안들이 서로 많이 겹치므로 **합성곱 연산이 중복**된다.

**SPPnet**(He et al., 2014)은 이미지 전체에 대해 conv 특징 맵을 한 번만 계산하고 각 제안 영역의 특징을 spatial pyramid pooling으로 뽑아
테스트를 크게 가속했다. 그러나 SPPnet도 R-CNN과 같은 다단계 학습이며, 특징을 디스크에 저장해야 하고,
**SPP 층 이전의 conv 층을 미세조정할 수 없다**(역전파가 SPP층을 지나 conv로 효율적으로 전달되지 못함 — 서로 다른 이미지에서 온 RoI가 배치에 섞이기 때문에 receptive field가 매우 크고 비효율적).
이는 매우 깊은 망에서 정확도 한계로 이어진다.

Fast R-CNN의 목표: **단일 단계(single-stage) 학습**, **모든 층을 업데이트 가능**, **저장 공간 불필요**, 그리고 R-CNN/SPPnet보다 더 빠르고 정확하게.

## 2. 핵심 아이디어

1. **이미지당 conv 1회**: 전체 이미지를 CNN에 한 번 통과시켜 conv 특징 맵을 만들고, 모든 RoI가 이 특징 맵을 공유한다.
2. **RoI Pooling 층**: 특징 맵 위의 임의 크기 RoI를 고정 크기($H \times W$, 예: 7×7)의 특징으로 변환한다. SPP의 단일 피라미드 레벨 특수 케이스이다.
3. **Multi-task loss**: 분류(softmax)와 박스 회귀(smooth L1)를 하나의 네트워크 안에서 **동시에 end-to-end 학습**한다. SVM과 별도 회귀기가 불필요해진다.
4. **효율적인 미니배치 샘플링**: 이미지 $N$장(예: 2장)에서 이미지당 $R/N$개(예: 64개)의 RoI를 샘플링한다. 같은 이미지 RoI들이 계산을 공유하므로
   $R=128$ 기준 R-CNN(서로 다른 128개 이미지에서 1개씩)보다 약 64배 저렴하다.
5. **Truncated SVD**로 FC 층을 압축해 추론을 가속한다.

## 3. 아키텍처/방법

### 3.1 전체 구조

```
입력 이미지 ─► CNN conv layers (전체 이미지 1회) ─► conv feature map
                                                       │
 Region proposals (Selective Search, 외부) ─► RoI 목록 ─┤
                                                       ▼
                                 RoI Pooling (RoI마다 7×7 고정 크기)
                                                       ▼
                                     FC6 ─► FC7 ─► RoI feature vector
                                                  ├─► FC (K+1) + softmax  : 클래스 확률 p
                                                  └─► FC (4K)            : 클래스별 박스 보정 t^k
```

- 네트워크 초기화: ImageNet 사전학습 모델(CaffeNet=AlexNet 계열 S, M(VGG_CNN_M_1024), L(VGG-16))에서 시작한다.
- 변환 3가지:
  1. 마지막 max pooling 층을 **RoI Pooling**으로 교체 ($H, W$는 첫 FC층과 호환되도록 설정, VGG-16에서는 7×7).
  2. 마지막 FC + softmax(1000-way)를 **두 개의 형제(sibling) 층**으로 교체: (K+1)-way softmax 분류, 클래스별 박스 회귀.
  3. 입력을 **이미지 리스트 + 각 이미지의 RoI 리스트** 두 가지로 받도록 변경.

### 3.2 RoI Pooling

특징 맵 상의 RoI $(r, c, h, w)$ (좌상단 좌표와 높이·너비)를 $H \times W$ 격자로 나누고(각 셀 크기 약 $h/H \times w/W$), 각 셀 안에서 **max pooling**한다.
채널마다 독립적으로 수행하며, RoI 좌표는 입력 이미지 좌표를 특징 맵 stride(예: VGG-16에서 16)로 나눠 양자화(반올림)해 얻는다.

역전파: 출력 $y_{rj}$가 입력 $x_i$에서 max를 취한 것이라면, 그래디언트는 argmax 위치로만 전달된다.

$$\frac{\partial L}{\partial x_i} = \sum_{r}\sum_{j} \left[ i = i^*(r, j) \right] \frac{\partial L}{\partial y_{rj}},\qquad i^*(r,j) = \arg\max_{i' \in \mathcal{R}(r,j)} x_{i'}$$

한 입력 위치가 여러 RoI 출력에서 선택될 수 있으므로 그래디언트를 합산한다. 이 덕분에 conv 층까지 역전파가 가능해진다.
(양자화로 인한 위치 어긋남은 이후 Mask R-CNN의 RoIAlign에서 개선된다.)

### 3.3 Multi-task loss

각 학습 RoI는 정답 클래스 $u$ (배경은 $u = 0$)와 정답 박스 회귀 타깃 $v$를 가진다.

$$L(p, u, t^u, v) = L_{cls}(p, u) + \lambda\,[u \ge 1]\, L_{loc}(t^u, v)$$

- $L_{cls}(p, u) = -\log p_u$ : 정답 클래스의 log loss.
- $[u \ge 1]$ : 배경 RoI는 박스 회귀 손실에서 제외(Iverson bracket).
- $L_{loc}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t^u_i - v_i)$

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5\,x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

- **smooth L1**은 L2 대비 이상치(outlier)에 덜 민감하다(그래디언트가 크기와 무관하게 상한). R-CNN/SPPnet에서 쓰던 L2 손실은 학습률을 신중히 조정하지 않으면 그래디언트 폭주가 생길 수 있었다.
- 회귀 타깃 $v$는 R-CNN과 같은 파라미터화(중심 오프셋은 제안 크기로 정규화, 너비·높이는 log 비율)를 쓰고, 평균 0 / 단위 분산이 되도록 **정규화**한다.
- 논문에서 $\lambda = 1$ 로 두었다.

### 3.4 미니배치 샘플링

| 항목 | 값 |
|---|---|
| 미니배치당 이미지 수 $N$ | 2 |
| 미니배치당 RoI 수 $R$ | 128 (이미지당 64) |
| 양성 RoI | 전체의 25% (≈16개/이미지), GT와 **IoU ≥ 0.5** → 해당 클래스 |
| 배경 RoI | 나머지, 최대 IoU가 **[0.1, 0.5)** 구간 (낮은 하한은 hard example mining의 휴리스틱 역할) |
| 데이터 증강 | 좌우 반전 (확률 0.5) |

R-CNN과 달리 양성/음성 기준이 분류 및 회귀에 통일되어 있다. 이미지당 RoI들이 conv를 공유하므로 효율적이다.
(RoI들이 같은 이미지에서 와서 상관되어 수렴이 느려질 우려가 있었으나 실제로는 문제되지 않았다고 논문은 보고한다.)

### 3.5 스케일 불변성

두 가지 방법을 비교했다: (1) brute-force — 학습·테스트 모두 고정 크기로 처리해 네트워크가 스케일 불변성을 직접 학습, (2) 이미지 피라미드.
논문은 깊은 망이 스케일 불변성을 직접 잘 학습하므로, 단일 스케일(짧은 변 600)이 속도-정확도 균형이 좋다고 결론지었다(피라미드는 mAP 소폭 향상, 비용은 큼).

### 3.6 Truncated SVD를 이용한 가속

탐지에서는 RoI가 많아 **FC층 계산 시간이 전체의 절반 가까이**를 차지한다. $u \times v$ 가중치 행렬 $W$를 SVD로 근사한다.

$$W \approx U\,\Sigma_t\,V^\top$$

- $U$: $u \times t$, $\Sigma_t$: $t \times t$, $V$: $v \times t$ 이면 파라미터는 $uv \to t(u+v)$ 로 줄어든다($t \ll \min(u, v)$일 때 이득).
- 하나의 FC층을 두 FC층(첫째: $\Sigma_t V^\top$, 둘째: $U$, 첫째에는 bias 없음)으로 대체한다.
- 논문은 VGG-16의 fc6(25088→4096)에서 상위 1024개, fc7(4096→4096)에서 상위 256개 특이값을 유지하면 mAP 손실이 매우 작으면서(약 0.3%p) 추론 속도를 크게(약 30% 가량) 높일 수 있다고 보고한다.

### 3.7 R-CNN, SPPnet과의 비교 (학습·테스트 구조)

| 항목 | R-CNN | SPPnet | Fast R-CNN |
|---|---|---|---|
| conv 계산 | 제안마다 | 이미지당 1회 | 이미지당 1회 |
| 학습 단계 | 3단계 (CNN, SVM, bbox reg) | 3단계 | **1단계 (multi-task)** |
| 특징 디스크 저장 | 필요 | 필요 | **불필요** |
| conv 층 미세조정 | 가능(하지만 느림) | 불가(SPP 이전 고정) | **가능** |
| 분류기 | 선형 SVM | 선형 SVM | **Softmax (+ 회귀 동시)** |

## 4. 실험 결과 요약

| 항목 | R-CNN (VGG-16) | SPPnet (VGG-16) | Fast R-CNN (VGG-16) |
|---|---|---|---|
| VOC07 mAP | 약 66% | 약 63% | 약 66.9% |
| 학습 시간 | 약 84시간 | 약 25시간 | 약 9.5시간 (R-CNN 대비 약 9배 빠름) |
| 테스트 속도 (이미지당) | 약 47초 | 약 2.3초 | 약 0.3초 (R-CNN 대비 약 146배, SVD 사용 시 약 213배 빠름) |

- PASCAL VOC 2012 test에서 약 65.7% mAP(VOC07+12 학습 시 더 높음; 이전 R-CNN 약 62%) 수준을 기록했다고 보고한다.
- **Softmax vs SVM**: Fast R-CNN의 softmax가 SVM보다 약간 좋거나 동등 → 별도 SVM 학습이 필요 없다는 근거.
- **Multi-task 학습 효과**: 분류만 학습하는 것보다 분류+회귀를 함께 학습하는 것이 mAP를 약 0.8~1.1%p 향상(multi-task가 서로 도움).
- **어느 층을 미세조정?**: VGG-16에서 conv3_1 이상을 미세조정하는 것이 중요하며, conv1 계열은 굳이 필요 없다(학습 시간 증가 대비 이득이 적음). FC만 미세조정하면 성능이 크게 낮다.
- **더 많은 제안이 항상 좋은 것은 아님**: 제안 수가 일정 이상이면 mAP가 오히려 하락할 수 있다. 밀집(dense) 박스는 성능이 더 낮다.
- **시간 분해**: Selective Search 제안 생성 시간(이미지당 약 2초)이 이제 전체 시간의 병목이 된다. → Faster R-CNN의 동기.

(수치는 논문 보고 값에 대한 개략적 서술이며, 정확한 값은 원문 표를 참고할 것.)

## 5. 의의 (앞뒤 논문과의 연결)

- **R-CNN → Fast R-CNN**: "conv 공유 + RoI Pooling + multi-task loss"로 R-CNN의 느림·복잡성을 해결하고 정확도도 향상시켰다.
  이후의 거의 모든 two-stage 탐지기(Faster R-CNN, FPN, Mask R-CNN)의 두 번째 단계(RoI head)가 사실상 Fast R-CNN 구조이다.
- **한계**: 영역 제안(Selective Search)이 CPU에서 별도로 돌아 병목이 되며 end-to-end가 아니다.
- **→ Faster R-CNN (다음 주)**: Region Proposal Network(RPN)로 제안 생성까지 CNN 안에 통합하고 conv 특징을 RPN과 Fast R-CNN 헤드가 공유한다.
- **→ Mask R-CNN**: RoI Pooling의 좌표 양자화 문제를 RoIAlign으로 해결하고, 마스크 분기를 추가한다.
- smooth L1 손실과 박스 파라미터화 방식은 이후 SSD, RetinaNet 등 많은 탐지기가 이어서 사용한다.

## 6. 구현 노트

`implementation/model.py`에 다음을 처음부터 구현했다.

- **RoIPool**: `torchvision.ops.roi_pool` 없이 순수 PyTorch로 구현(각 RoI를 stride로 나눠 반올림 후 H×W 격자로 분할, 셀마다 max). 학습 가능한 그래디언트가 argmax 위치로 흐르는지 확인 가능.
- **FastRCNN**: 소형 conv 백본(feature stride 16) + RoIPool(7×7) + FC6/FC7 + (K+1) 분류 헤드 + 4K 회귀 헤드.
- **fast_rcnn_loss**: 분류 cross-entropy + 양성 RoI에 대한 smooth L1 multi-task loss.
- **sample_rois**: 이미지당 64 RoI, 양성 25%(IoU ≥ 0.5), 배경 IoU [0.1, 0.5) 샘플링.
- **TruncatedLinear / truncate_linear**: FC를 SVD로 두 층으로 분해하는 압축 예시.
- **현대적 단순화**: 논문의 ImageNet 사전학습 VGG-16/CaffeNet 대신 학습 가능한 작은 conv 백본을 사용(구조 이해 목적). 박스 타깃의 평균/표준편차 정규화는 생략 가능(옵션으로 `box_std` 제공).
  Selective Search는 구현하지 않고 외부 제안 박스를 입력으로 받는다.
- `train.py`: 데이터셋이 무거우므로 **더미 이미지/GT/제안**으로 RoI 샘플링 → loss → 1 step 학습을 시연한다.
