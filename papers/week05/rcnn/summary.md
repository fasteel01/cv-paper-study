# Week 5-1. R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation**
Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik — *CVPR*, 2014 (arXiv 1311.2524)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

**객체 탐지(object detection)**는 이미지 분류와 달리 "무엇이 있는가"뿐 아니라 "어디에 있는가"(bounding box)까지 맞혀야 한다.
2012년 AlexNet이 ImageNet 분류에서 큰 성공을 거두자 자연스러운 질문이 생겼다.
**"ImageNet 분류에서의 CNN 성능 향상이 PASCAL VOC 같은 객체 탐지로도 일반화되는가?"**

당시 상황은 다음과 같았다.

- PASCAL VOC 탐지 성능은 2010~2012년 사이 거의 정체 상태였다. 최고 성능 방법들은 SIFT, HOG 같은 **손으로 설계한 특징** +
  DPM(Deformable Part Model)류의 복잡한 앙상블 시스템이었다.
- 탐지용 **라벨 데이터(VOC)는 적다**(수천~1만 장 수준). 큰 CNN을 처음부터 학습하기에는 부족하다.
- 분류와 달리 탐지는 **위치 추정**이 필요하다. CNN으로 이를 어떻게 풀 것인가?
  - 위치를 직접 회귀(regression)하는 방식은 당시 성능이 좋지 않았다.
  - 슬라이딩 윈도우 방식은 층이 깊은 CNN에서는 수용 영역(receptive field)과 stride가 커서 정확한 위치 추정이 어렵다.

R-CNN은 이 두 문제(**적은 라벨 데이터**, **위치 추정**)를 다음 두 통찰로 해결한다.

1. **"Recognition using regions"**: 지역 후보(region proposal)를 먼저 만들고, 각 후보를 CNN으로 분류한다.
2. **감독 사전학습 + 도메인 특화 미세조정(fine-tuning)**: 라벨이 풍부한 보조 과제(ImageNet 분류)로 사전학습한 뒤,
   라벨이 적은 목표 과제(탐지)로 미세조정한다.

## 2. 핵심 아이디어

R-CNN = **R**egions with **CNN** features. 파이프라인은 3개 모듈로 구성된다.

1. **영역 제안(Region proposal)**: 클래스와 무관하게 이미지당 약 2000개의 후보 박스를 생성한다. 논문은
   **Selective Search**(색·질감·크기·겹침 유사도 기반 계층적 영역 병합)를 사용한다.
2. **CNN 특징 추출**: 각 후보 영역을 고정 크기(227×227)로 **왜곡(warp)**해서 CNN에 넣어 4096차원 특징 벡터를 얻는다.
3. **클래스별 선형 SVM**: 4096차원 특징을 클래스마다 하나씩 학습된 선형 SVM으로 점수화한다.
   이후 클래스별 **NMS(Non-Maximum Suppression)**로 중복 박스를 제거한다.

추가로 **Bounding-box regression**으로 제안 박스 위치를 보정해 localization 오류를 줄인다.

핵심 학습 전략은 다음과 같다.

- **감독 사전학습(supervised pre-training)**: ImageNet 분류(ILSVRC2012)로 CNN을 학습한다(탐지 박스 라벨 없이 이미지 단위 라벨만 사용).
- **도메인 특화 미세조정(domain-specific fine-tuning)**: 마지막 1000-way 분류층을 (N+1)-way(N개 클래스 + 배경)로 교체하고,
  warp된 제안 영역들로 SGD 미세조정한다.
- 이후 CNN은 **특징 추출기**로만 쓰고 그 위에 SVM을 별도로 학습한다(end-to-end가 아님 → 이후 Fast/Faster R-CNN에서 개선).

## 3. 아키텍처/방법

### 3.1 전체 파이프라인

```
입력 이미지
  └─ Selective Search ──► 약 2000개 region proposal
        └─ 각 영역을 227×227로 warp (context 16px 포함)
              └─ CNN (AlexNet 계열, conv×5 + fc×2) ──► 4096-d 특징
                    ├─ 클래스별 선형 SVM ──► 클래스 점수
                    └─ 클래스별 bbox regressor (pool5 특징 사용) ──► 박스 보정
                          └─ 클래스별 NMS ──► 최종 탐지
```

### 3.2 영역 전처리 (warping)

- CNN은 고정 크기 입력을 요구하므로, 후보 박스의 종횡비와 무관하게 **타이트 박스 주변 16픽셀의 문맥(context)을 포함**해
  227×227로 **비등방(anisotropic) 리사이즈**한다. 영역이 이미지 밖으로 나가는 부분은 평균 값으로 채운다.
- 논문은 비교 실험으로 context 없이 타이트하게 자르는 방식, 정사각형 패딩 등과 비교했고, warp + context(p=16)가 가장 좋았다고 보고한다.

### 3.3 CNN 구조

- 기본: AlexNet(Krizhevsky et al.)과 동일한 구조 — conv1~conv5, fc6, fc7. 4096차원 fc7 출력을 특징으로 쓴다.
- 이후 실험에서 VGG-16 계열("O-Net" = OxfordNet)로 교체하면 성능이 크게 향상되지만 계산량이 매우 늘어난다.
  (T-Net = TorontoNet = AlexNet.)

### 3.4 학습 단계

| 단계 | 데이터 | 라벨/양성 기준 | 비고 |
|---|---|---|---|
| ① 사전학습 | ImageNet 분류 | 이미지 단위 클래스 | 1000-way 분류 |
| ② 미세조정 | VOC 제안 영역 (warp) | **IoU ≥ 0.5** 이면 해당 GT 클래스, 그 외 배경 | 분류층을 (N+1)-way로 교체. 미니배치 128 = 양성 32 + 배경 96 (양성에 편향) |
| ③ SVM 학습 | fc7 특징 | **양성 = GT 박스 자체**, **음성 = 모든 GT와 IoU < 0.3** | IoU 0.3~0.5 구간은 버림. hard negative mining 사용 |
| ④ Bbox 회귀 | pool5 특징 | 가장 가까운 GT와 IoU > 0.6 인 제안 | 클래스별 릿지(ridge) 회귀 |

여기서 눈여겨볼 점은 **미세조정(IoU ≥ 0.5)**과 **SVM(양성은 GT만, 음성은 IoU < 0.3)**이 서로 다른 기준을 쓴다는 것이다.
논문은 미세조정 시 위치가 부정확한 예제까지 양성으로 쓰는 것이 과적합을 줄이는 데 도움되지만, 그 결과 CNN 특징은 정확한 localization에
최적화되지 않아 SVM을 따로 학습하는 것이 더 낫다고 설명한다. (이 이중 구조는 복잡하고 느리며, Fast R-CNN에서 단일 softmax 분류로 통합된다.)

### 3.5 IoU (Intersection over Union)

두 박스 $A$, $B$에 대해

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

탐지에서 양성/음성 판정, NMS, 평가(mAP)의 기본 척도이다.

### 3.6 Bounding-box regression

제안 박스 $P = (P_x, P_y, P_w, P_h)$(중심 좌표 + 너비·높이)를 정답 박스 $G = (G_x, G_y, G_w, G_h)$로 옮기는 변환을 학습한다.
회귀 타깃은 스케일 불변이 되도록 다음과 같이 정의한다.

$$t_x = \frac{G_x - P_x}{P_w},\quad t_y = \frac{G_y - P_y}{P_h},\quad t_w = \log\frac{G_w}{P_w},\quad t_h = \log\frac{G_h}{P_h}$$

각 클래스마다 pool5 특징 $\phi_5(P)$의 선형 함수 $d_\star(P) = \mathbf{w}_\star^\top \phi_5(P)$ ($\star \in \{x,y,w,h\}$)를 릿지 회귀로 학습한다.

$$\mathbf{w}_\star = \arg\min_{\hat{\mathbf{w}}} \sum_i \left(t^i_\star - \hat{\mathbf{w}}^\top \phi_5(P^i)\right)^2 + \lambda \|\hat{\mathbf{w}}\|^2$$

추론 시에는 다음과 같이 박스를 보정한다.

$$\hat G_x = P_w d_x(P) + P_x,\quad \hat G_y = P_h d_y(P) + P_y,\quad \hat G_w = P_w \exp(d_w(P)),\quad \hat G_h = P_h \exp(d_h(P))$$

(논문은 $\lambda$를 1000으로 설정했다고 기술한다. 제안이 GT에 가까울 때(IoU가 충분히 클 때)만 학습해야 선형 근사가 성립한다.)

### 3.7 NMS (Non-Maximum Suppression)

클래스별로 점수 내림차순 정렬 후, 가장 높은 박스와 IoU가 임계값(논문은 학습된 임계값 사용)을 넘는 나머지 박스를 제거하는 과정을 반복한다.

```
1. 점수 순 정렬
2. 최고점 박스를 결과에 추가
3. 그 박스와 IoU > τ 인 나머지 박스 제거
4. 박스가 남아 있으면 2로
```

### 3.8 추론 비용

- 이미지당 약 2000개의 영역을 **각각 독립적으로 CNN에 통과**시킨다 → 이미지 한 장당 GPU에서 수십 초(논문: VGG-16 기준 약 47초/이미지, AlexNet 기준 약 13초/이미지로 보고).
- 특징을 디스크에 저장한 뒤 SVM을 학습하므로 저장 공간(수백 GB 규모)과 시간이 크게 소요된다.

## 4. 실험 결과 요약

| 데이터셋 | 방법 | mAP |
|---|---|---|
| PASCAL VOC 2010 | 이전 최고(Selective Search + 다중 특징 기반 시스템 UVA 등) | 약 35% |
| PASCAL VOC 2010 | R-CNN (AlexNet, fc7) | 약 50% |
| PASCAL VOC 2010 | R-CNN + bbox regression (AlexNet) | 약 53% |
| PASCAL VOC 2010 | R-CNN + bbox regression (VGG-16) | 약 62% |
| ILSVRC2013 detection | OverFeat(당시 최고) | 약 24% |
| ILSVRC2013 detection | R-CNN + bbox regression | 약 31% |

주요 관찰:

- 사전학습만 하고 미세조정하지 않은 경우, **fc6/fc7이 pool5보다 오히려 성능이 비슷하거나 낮음** — 즉 conv 특징 자체가 일반적이고 FC층은 도메인 특화 정보를 담는다.
- **미세조정을 하면 fc6/fc7에서 큰 성능 향상**(mAP 수 %p 이상)이 나타난다. 특히 FC층에서 이득이 크다.
- bbox regression은 mAP를 약 3~4%p 향상시키며, 주된 오류 원인이 **mislocalization**임을 보여준다.
- 오류 분석(Hoiem et al.의 도구)에서 가장 많은 오류는 위치 부정확(Loc)이며, 이것이 bbox regression으로 상당 부분 감소한다.
- 같은 CNN 특징으로 **semantic segmentation**(VOC 2011 segmentation)에도 적용 가능함을 보였다(부제의 "semantic segmentation").

(수치는 논문 보고 값에 대한 개략적인 서술이며, 정확한 값은 원문 표를 참고할 것.)

## 5. 의의 (앞뒤 논문과의 연결)

- **AlexNet → R-CNN**: ImageNet에서 학습한 CNN 특징이 탐지에도 전이(transfer)됨을 보인 첫 대표 사례. "사전학습 + 미세조정"이라는
  현재의 표준 레시피를 컴퓨터 비전 전반에 확립했다.
- **한계**:
  1. 영역마다 CNN을 따로 실행 → 극도로 느림(합성곱 계산을 중복 수행).
  2. 3단계 학습(CNN 미세조정 → SVM → bbox regressor)으로 복잡하고, 특징을 디스크에 저장해야 함.
  3. Selective Search는 CPU 기반의 고정된 알고리즘이라 학습되지 않는다.
- **→ Fast R-CNN (다음 논문)**: 이미지를 한 번만 CNN에 통과시키고 RoI Pooling으로 영역별 특징을 뽑으며, 분류와 박스 회귀를 한 네트워크에서 동시에 학습한다.
- **→ Faster R-CNN**: Selective Search까지 신경망(RPN)으로 대체한다.
- 같은 시기 SPPnet(He et al., 2014)이 conv 계산 공유 아이디어를 제안했고 이는 Fast R-CNN의 동기가 된다.

## 6. 구현 노트

`implementation/model.py`에는 R-CNN의 구성요소를 PyTorch로 처음부터 구현했다.

- **포함**: IoU 계산, 박스 인코딩/디코딩(bbox regression 타깃), 영역 warp(crop → 227×227 리사이즈, context 패딩), 클래스별 NMS,
  AlexNet 계열 특징 추출기(fc7 4096-d), 선형 분류 헤드, 박스 회귀 헤드, 추론 파이프라인.
- **생략/단순화**: Selective Search는 구현하지 않고 외부에서 박스를 받는다(더미 박스로 시연). 클래스별 SVM + hard negative mining 대신
  미세조정된 CNN의 **선형 분류층(softmax)**을 사용하는 현대적 단순화를 적용했다(Fast R-CNN 논문이 보인 방식). bbox regression도
  pool5 릿지 회귀 대신 **fc7 위의 선형층 + smooth L1**로 구성했다.
- `train.py`: 탐지 데이터셋(VOC)은 무거우므로 **더미 이미지/박스로** 제안 라벨링(IoU 기준), 샘플링(양성 32 / 배경 96), 분류+회귀 손실 계산, 1 step 학습을 보여준다.
