# Week 9-2. Mask R-CNN

**Mask R-CNN**
Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick — *ICCV*, 2017 (arXiv 1703.06870)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

**인스턴스 분할(instance segmentation)** = 객체 검출(각 객체 박스·클래스) + 의미 분할(픽셀 단위). "같은 클래스의 서로 다른 개체"를 구분해 각각의 픽셀 마스크를 내야 한다.

이전 접근:
- **Segment proposal 기반**(SDS, DeepMask → MNC, FCIS): 분할이 분류에 선행하거나, 여러 단계(multi-stage)를 거치며 느림. FCIS 는 위치 민감 점수맵(position-sensitive)을 써서 겹치는 객체에서 이상한 아티팩트를 보임.
- 검출 쪽에서는 Faster R-CNN(+FPN)이 강력한 틀을 제공했으나 **박스만** 예측.

문제의식: *Faster R-CNN 에 "병렬 분기" 하나만 추가해, 단순하고 유연하게 인스턴스 마스크를 얻을 수 있을까?* 병목은 RoIPool 이었다 — 정수 좌표로 양자화(quantization)하기 때문에 **박스 분류에는 무관하나 픽셀 단위 정렬이 필요한 마스크에서는 치명적**이다.

## 2. 핵심 아이디어

1. **마스크 분기 추가**: Faster R-CNN 의 분류·박스 회귀 head 와 **병렬**로, 각 RoI 에 대해 작은 FCN 이 $K$개 클래스별 $m\times m$ 이진 마스크를 예측. 학습·추론 모두 단순.
2. **RoIAlign**: RoIPool 의 두 번의 반올림(RoI 좌표 → 특징맵 격자, RoI → 빈(bin))을 없애고, 각 bin 안의 몇 개 샘플링 점에서 **bilinear 보간** 으로 특징값을 구한 뒤 평균/최댓값. 정확한 공간 정렬을 보존.
3. **마스크와 클래스 예측의 분리(decoupling)**: 마스크 손실은 **정답 클래스 채널 하나**에만 적용되는 픽셀별 **sigmoid + 이진 교차 엔트로피**. 클래스 간 softmax 경쟁이 없어, 클래스 판단은 분류 분기에 맡기고 마스크 분기는 "이 클래스의 모양" 만 학습.
4. **다중 과제 학습**: $L = L_{cls} + L_{box} + L_{mask}$ (RPN 손실 별도). 마스크 학습이 박스 성능도 소폭 높여주는 효과(동시 학습의 이득)가 관찰됨.
5. **일반 프레임워크**: 같은 구조로 **인체 자세(keypoint)** 추정까지 확장(키포인트마다 one-hot 마스크 예측).

## 3. 아키텍처/방법

### 3.1 전체 파이프라인

```
이미지 → 백본(ResNet-50/101 또는 ResNeXt, + FPN) → RPN(영역 제안)
       → 각 RoI 에 RoIAlign → ┬ 분류 + 박스 회귀 head (FC)
                               └ 마스크 head (FCN, 클래스별 m×m)
```

### 3.2 RoIAlign

- RoIPool: RoI 좌표 $x/16$ 을 **반올림**, 다시 bin 으로 나눌 때 또 반올림 → 특징과 원본 픽셀 사이 오정렬(최대 수 픽셀, stride 32 에서 치명적).
- RoIAlign: 부동소수점 좌표를 그대로 유지. RoI 를 $7\times7$ (또는 $14\times14$) bin 으로 나누고, 각 bin 에서 규칙적인 4개 위치(2×2)에 bilinear 보간으로 값을 샘플한 뒤 평균(또는 max) → 결과는 샘플링 위치와 개수에 거의 민감하지 않다.
- FPN 사용 시 RoI 를 크기로 레벨에 할당: $k = \lfloor k_0 + \log_2(\sqrt{wh}/224)\rfloor$, $k_0=4$, $k\in[2,5]$.

### 3.3 헤드 구조

| 헤드 | 구성 (FPN 버전) | 출력 |
|---|---|---|
| 박스 head | RoIAlign **7×7×256** → FC 1024 → FC 1024 | 클래스 점수 $K{+}1$, 클래스별 박스 $4K$ |
| 마스크 head | RoIAlign **14×14×256** → conv3×3(256) ×4 → deconv 2×2 stride 2(256) → conv1×1($K$) | **28×28 × $K$** 마스크 로짓 |

- 원래(C4 백본) 버전은 ResNet 의 res5 를 head 로 사용. FPN 버전은 위처럼 가볍고 더 빠르다.
- 마스크 head 는 **FC 가 아닌 conv** 로 공간 구조(m×m)를 유지 → 파라미터가 적고 정확. 논문 ablation 에서 FCN 형태 head 가 MLP 보다 우수.

### 3.4 손실

- $L_{cls}$: 분류 교차 엔트로피, $L_{box}$: smooth L1 (Fast R-CNN 과 동일).
- **마스크 손실**: 양성 RoI(정답 IoU ≥ 0.5)에서, 정답 클래스가 $k$ 라면 $k$번째 $28\times28$ 마스크만 사용:

$$L_{mask} = -\frac{1}{m^2}\sum_{i,j}\big[y_{ij}\log\sigma(\hat{y}^{k}_{ij}) + (1-y_{ij})\log(1-\sigma(\hat{y}^{k}_{ij}))\big]$$

- 정답 마스크 $y$ 는 이미지 크기의 마스크를 RoI 영역에 맞춰 $m\times m$ 로 잘라 리사이즈(RoIAlign 과 동일 좌표계)한 것.
- 전체: $L = L_{cls} + L_{box} + L_{mask}$ (가중치 1:1:1), RPN 손실은 별도.

### 3.5 학습 · 추론 디테일

- 학습: RoI 샘플링 이미지당 512개(양성 25%), 양성은 IoU ≥ 0.5 이며 마스크 손실은 양성에만. 짧은 변 **800 px** 로 리사이즈, 배치 16(8 GPU × 2장), SGD momentum 0.9, weight decay 1e-4, 초기 lr 0.02, 약 160k iteration(중간에 ×0.1).
- 추론: 박스 head 로 먼저 검출 → NMS → **상위 100개 검출에 대해서만 마스크 head 실행**(약간의 연산 추가) → 선택된 클래스의 마스크를 sigmoid → 임계값 0.5 로 이진화, 박스 크기로 리사이즈해 이미지에 붙임.

## 4. 실험 결과 요약

(논문 보고값 기준, 약 값)

- **COCO test-dev 인스턴스 분할**: ResNet-101-FPN 마스크 AP 약 **35.7**, ResNeXt-101-FPN 약 **37.1**. 당시 최고였던 MNC·FCIS(+++)를 앞섬(FCIS+++ 약 33.6 → 개선), 겹치는 객체에서의 아티팩트도 없음.
- **박스 검출**: 마스크 분기를 학습하면 박스 AP 도 소폭 상승 (ResNeXt-101-FPN 에서 박스 AP 약 39~40, 당시 단일 모델 최고 수준).
- **속도**: 약 5 fps (Tesla M40 기준, 백본에 따라 상이).
- **Ablation** (ResNet-50 기반):
  - *Decoupling*: 클래스 간 softmax 마스크 대신 클래스별 sigmoid → 마스크 AP **약 5.5p 향상**.
  - *RoIAlign vs RoIPool*: AP 향상이 있고 특히 **높은 IoU(AP75)** 에서 큼, stride 가 큰 백본(C5)에서 더 극적.
  - *Head 구조*: FCN 마스크 head 가 MLP 보다 우수. 다중 과제 학습이 서로를 도움.
- **Cityscapes** 인스턴스 분할에서도 당시 우수한 결과(AP 약 26). **인체 키포인트**(COCO) AP 약 63 으로 당시 1위권(마스크 분기 구조를 그대로 재활용).

## 5. 의의 (앞뒤 논문과의 연결)

- **앞 (R-CNN → Fast → Faster R-CNN)**: "영역 제안 + RoI 별 head" 의 정점. Faster R-CNN + FPN 에 병렬 head 하나를 추가한 단순한 설계가 곧 강력한 기준선(baseline)이 됨.
- **앞 (FPN)**: 다중 스케일 RoI 할당과 P2~P6 이 그대로 사용됨. 작은 객체 마스크 성능에 필수.
- **앞 (FCN)**: 마스크 head 는 RoI 단위의 작은 FCN ("mask branch = FCN"). FCN 의 "클래스 간 softmax 경쟁" 대신 sigmoid 로 분리했다는 점이 차이.
- **앞 (RetinaNet / YOLO / SSD)**: 1단계 검출기가 속도를 택한 반면, Mask R-CNN 은 2단계의 정확도를 인스턴스 분할로 확장.
- **옆 (U-Net)**: 같은 주의 다른 길 — U-Net 은 세미/바이오 의료 분할의 밀집 예측, Mask R-CNN 은 객체 중심의 인스턴스 분할.
- **뒤 (DeepLabv3)**: 다음 주, 의미 분할을 atrous conv 로 다시 정교화.
- **뒤 (DETR / Mask2Former)**: DETR 는 NMS 와 앵커를 없애고 Transformer 로 박스 집합을 직접 예측하며, 이를 분할로 확장한 DETR-panoptic, Mask2Former 등이 Mask R-CNN 의 자리를 이어받음. **Segment Anything** 같은 최신 모델도 "프롬프트 → 마스크 디코더" 로 이 계보를 잇는다.

## 6. 구현 노트

`implementation/model.py` 에 다음을 구현했다 (RoIAlign 연산과 NMS 만 `torchvision.ops` 사용).
- ResNet-50 + FPN(P2~P5, P6 = P5 의 stride-2 서브샘플). BN 은 일반 BN(공식은 frozen BN), 사전학습 없음.
- 앵커 생성(레벨당 1 크기 × 3 종횡비) · RPN(head 가중치 공유, 레벨별 top-k + 레벨별 NMS 제안, IoU 0.7/0.3 매칭, 256개 샘플링, BCE + smooth L1).
- `MultiScaleRoIAlign`: 식 $k=\lfloor4+\log_2(\sqrt{wh}/224)\rfloor$ 로 레벨 할당 후 `roi_align(aligned=True)`.
- Box head(FC1024×2, 클래스별 회귀, 가중치 (10,10,5,5) 인코딩), Mask head(conv×4 + deconv + 1×1 conv, $K$ 채널).
- **마스크 손실**: 양성 RoI 의 정답 클래스 채널만 골라 sigmoid BCE, 정답 마스크는 `roi_align` 으로 28×28 타깃 생성.
- 추론: 클래스별 박스 디코딩 → 점수 임계값 → 클래스별 NMS → 검출 박스에 대해서만 마스크 head → `paste_mask` 로 원본 크기에 복원.
- `__main__`: FPN shape, 5개 손실 계산 + backward, RoIAlign/헤드 출력 shape(`(R,K,28,28)`), 추론 출력 형태를 확인. 파라미터 수는 K=4 기준 약 44M(COCO 81 클래스에서도 약 44M 규모).

`implementation/train.py`: 합성 인스턴스(사각형/원) 2장에 대해 5개 손실을 합쳐 20 step 학습하고 마스크 IoU 계산 코드를 보여준다 (사전학습 없는 짧은 학습이라 마스크 품질은 낮음; 파이프라인 확인용).
