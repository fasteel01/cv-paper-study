# Week 8-2. FCN

**Fully Convolutional Networks for Semantic Segmentation**
Jonathan Long, Evan Shelhamer, Trevor Darrell — *CVPR*, 2015 (arXiv 1411.4038)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

**Semantic segmentation**은 이미지의 모든 픽셀에 클래스 라벨을 부여하는 문제다 (분류 = 이미지 1개당 라벨 1개, 검출 = 박스, 분할 = 픽셀 단위).

FCN 이전의 접근:
- 픽셀(또는 슈퍼픽셀) 주변의 패치를 잘라 CNN으로 분류 → 겹치는 패치마다 계산을 반복해 매우 비효율적이고, 패치 크기가 문맥 범위를 제한.
- R-CNN류의 SDS(Simultaneous Detection and Segmentation) 등: 영역 제안에 의존하는 다단계 파이프라인.
- 분류용 CNN(AlexNet, VGG, GoogLeNet)은 마지막에 **FC 층**이 있어 입력 크기가 고정되고 공간 정보가 사라진다.

문제의식: *분류망을 그대로 "픽셀 단위 예측기"로 바꿔, 임의 크기의 이미지를 입력받아 종단간(end-to-end) 학습할 수는 없을까?*

## 2. 핵심 아이디어

1. **Fully convolutional화**: FC 층을 커널이 입력 전체 크기(예: 7×7)인 합성곱으로 해석. 그러면 네트워크가 conv/pool만으로 이루어져 **임의 크기 입력에 대해 공간적으로 대응하는 출력 맵(히트맵)** 을 낸다. 슬라이딩 윈도우를 한 번의 forward로 공유 계산하는 것과 같다.
2. **전이 학습(fine-tuning)**: ImageNet으로 사전학습한 분류망(AlexNet, VGG-16, GoogLeNet)을 분할용으로 이식해 미세 조정.
3. **학습 가능한 업샘플링(전치 합성곱, "deconvolution")**: 저해상도 score 맵을 입력 해상도로 복원. bilinear 보간 가중치로 초기화하고 학습.
4. **Skip architecture**: 깊은 층(의미는 강하나 거친 해상도)과 얕은 층(해상도 높고 정밀하나 의미 약함)의 예측을 합쳐 *what*(의미)과 *where*(위치)를 결합. FCN-32s → 16s → 8s.
5. **픽셀 단위 손실**: 최종 출력 맵의 모든 픽셀에 대한 교차 엔트로피 합(=각 픽셀을 하나의 학습 샘플로 간주).

## 3. 아키텍처/방법

### 3.1 FC 층의 합성곱화

VGG-16 기준: pool5 출력은 $512\times7\times7$ (224 입력).

| 원래 | FCN | 비고 |
|---|---|---|
| fc6: Linear($512\cdot7\cdot7 \to 4096$) | Conv 7×7, 4096 채널 | 가중치를 $4096\times512\times7\times7$ 로 reshape하면 동일 |
| fc7: Linear($4096\to4096$) | Conv 1×1, 4096 | |
| fc8: Linear($4096\to1000$) | **score층**: Conv 1×1, $K$ 채널 ($K$=21: VOC 20 + 배경) | 새로 초기화 |

더 큰 입력(예: 500×500)에서는 pool5 맵이 더 커지고 fc6 conv가 슬라이딩하면서 **공간 격자 모양의 class score 맵**이 나온다. 이 출력은 입력 대비 stride 32로 거칠다.

### 3.2 업샘플링: 전치 합성곱

stride $s$ 배 업샘플은 커널 $2s$, stride $s$, padding $s/2$ 의 전치 합성곱으로 구현 (bilinear 초기화). 해상도를 입력 크기로 복원하면서 필터는 학습 가능.
논문은 이 방식이 고정 bilinear보다 약간 유리하거나 비슷함을 보고하고, 후속 연구에서는 학습된 필터가 큰 이득이 없다고 하여 고정 업샘플로 단순화한 변형도 많다.

### 3.3 Skip 구조 (FCN-32s / 16s / 8s)

| 모델 | 구성 |
|---|---|
| **FCN-32s** | pool5 위의 score를 32배 업샘플. 의미는 좋으나 경계가 뭉개짐 |
| **FCN-16s** | score를 2배 업샘플 + pool4 위 1×1 score(stride 16)를 **합산** → 16배 업샘플 |
| **FCN-8s** | 위 결과를 2배 업샘플 + pool3 위 1×1 score(stride 8) 합산 → 8배 업샘플 |

$$\hat{y}_{8s} = \text{Up}_{8}\Big(\text{Up}_{2}\big(\text{Up}_{2}(s_{5}) + s_{4}\big) + s_{3}\Big)$$

여기서 $s_l$ 은 $l$번째 pool 위에 붙인 1×1 conv score. 합산 전에 크기가 어긋나면 **crop**으로 맞춘다 (논문은 conv1_1에 padding 100을 줘 어떤 크기든 처리하고 crop 층을 사용).
skip score 층은 **0으로 초기화**하여 이전 단계의 예측을 해치지 않고 시작한다. 또한 32s → 16s → 8s를 단계적으로 이어서 학습한다. 논문은 pool2 이하로 더 내려가도 이득이 작다고 보고.

### 3.4 학습 디테일

- 손실: 픽셀별 softmax 교차 엔트로피의 합(또는 평균), 라벨이 없는 픽셀(VOC의 255 경계 영역)은 무시.
- 이미지 단위 학습으로 충분 — 패치 샘플링이나 클래스 균형 조정이 필요 없음을 보임.
- SGD + momentum 0.9, 작은 학습률, 큰 weight decay, 클래스 점수층 0 초기화, dropout 유지.

## 4. 실험 결과 요약

(논문 보고값 기준, 약 값. PASCAL VOC)

- **VOC 2011/2012 test mean IoU**: FCN-8s 약 62%. 당시 최고였던 SDS(약 51~52%) 대비 약 **20% 상대 향상**, 추론도 훨씬 빠름 (이미지당 약 0.2초 수준 vs. 수십 초).
- **Skip 효과 (VOC2011 val)**: 32s → 16s → 8s로 갈수록 mean IoU 상승(약 59 → 62 근방), 경계 디테일이 개선되는 정성 결과.
- **백본 비교**: VGG-16 기반이 AlexNet, GoogLeNet 기반보다 우수.
- **그 밖의 데이터셋**: NYUDv2(RGB-D), SIFT Flow(기하+의미 라벨) 등에서도 당시 SOTA 수준을 보임.

## 5. 의의 (앞뒤 논문과의 연결)

- **앞 (분류 CNN: AlexNet/VGG/GoogLeNet)**: "분류 백본 + 전이 학습"의 패러다임을 dense prediction으로 확장. FC 층이 "큰 커널 conv"라는 해석이 핵심.
- **앞 (R-CNN → Faster R-CNN → FPN, RetinaNet)**: 검출 계열이 "영역 제안 + 분류"라면 FCN은 "전체 이미지를 한 번에 픽셀 단위로". 동시에 FPN의 top-down + lateral 구조는 FCN skip 구조의 후손으로 볼 수 있다 (합산 방식의 융합, 얕은 층의 위치 정보 + 깊은 층의 의미 정보).
- **앞 (RetinaNet)**: 분할 역시 픽셀 단위 클래스 불균형이 심하며(배경 비율), Focal Loss가 분할에도 이식된다.
- **뒤 (U-Net)**: 다음 주 U-Net은 FCN의 skip을 확장해 **대칭형 인코더-디코더 + 채널 concat**, 디코더에도 풍부한 채널을 둔 구조.
- **뒤 (Mask R-CNN)**: RoI별 마스크 예측 분기가 작은 FCN. (논문도 "mask branch = FCN"임을 강조)
- **뒤 (DeepLab 계열)**: FCN의 거친 출력/작은 수용 영역 문제를 atrous 합성곱, CRF, 피라미드 풀링으로 개선.

## 6. 구현 노트

`implementation/model.py` 에 다음을 구현했다.
- VGG-16 conv 블록을 직접 쌓음 (pool3/pool4/pool5 출력 보관)
- fc6 = 7×7 conv, fc7 = 1×1 conv (+ Dropout2d), `score_fr`: 1×1 conv
- `bilinear_kernel` / `make_upsample`: bilinear로 초기화된 `ConvTranspose2d` (kernel $2s$, stride $s$, padding $s/2$)
- `stage` 인자로 FCN-32s / 16s / 8s 선택, skip 합산, 크기 불일치 시 중앙 crop(또는 bilinear)로 보정
- `__main__`: 세 변형 모두 `(N, 21, H, W)` 출력, 32의 배수가 아닌 입력 크기도 처리됨을 확인하고, 전치 합성곱이 bilinear 보간과 일치하는지 검증

원 논문과의 차이(자세한 내용은 model.py docstring): conv1_1 padding 100 대신 same-padding + crop/보간, 사전학습 가중치 없음, 변형별 처음부터 학습.
`train.py` 는 합성 데이터(사각형/원)로 픽셀 단위 CE 학습, 픽셀 정확도와 mean IoU(혼동 행렬)를 계산한다. 합성 데이터이므로 수치는 논문과 비교할 수 없다.
