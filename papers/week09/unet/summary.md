# Week 9-1. U-Net

**U-Net: Convolutional Networks for Biomedical Image Segmentation**
Olaf Ronneberger, Philipp Fischer, Thomas Brox — *MICCAI*, 2015 (arXiv 1505.04597)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

- 생체의학 영상(전자현미경 뉴런 구조, 세포 추적 등) 분할은 **픽셀 단위의 정밀한 경계**가 필요하고, 학습 데이터(전문가가 라벨링한 이미지)는 **수십 장 수준**으로 매우 적다.
- 기존 최고 성능 접근(Ciresan et al., 2012): 픽셀 주변 **패치를 잘라 슬라이딩 윈도우로 분류**하는 CNN. 단점은 (1) 겹치는 패치마다 계산 반복 → 느림, (2) 패치 크기에 따라 **문맥(context) 범위 vs 위치 정확도(localization)** 가 트레이드오프 (큰 패치 = 풀링이 많아 위치 정밀도 저하, 작은 패치 = 문맥 부족).
- FCN(전 주 논문)이 "분류망을 fully convolutional 로 바꿔 종단간 분할"을 제시했지만, 업샘플 경로가 얕아(대부분 하나의 큰 deconv) 경계가 거칠고, 적은 데이터에서의 학습 요령은 다루지 않았다.

문제의식: *아주 적은 라벨 데이터로, 문맥과 정밀한 위치를 동시에 얻는 종단간 분할망을 만들 수 없을까?*

## 2. 핵심 아이디어

1. **대칭형 인코더-디코더("U" 모양)**: 수축 경로(contracting path)로 문맥을 포착하고, 대칭인 확장 경로(expansive path)로 해상도를 단계적으로 복원.
2. **디코더에도 많은 채널**: 업샘플 경로에서도 특징 채널 수를 크게 유지(해상도가 오를 때 채널을 절반으로) → 문맥 정보가 고해상도 층까지 전달된다. FCN 은 업샘플 쪽 채널이 클래스 수(점수)뿐.
3. **채널 방향 concatenation skip**: 인코더의 같은 해상도 특징맵을 디코더 특징맵과 **이어붙임**(FCN: 점수 합산). 후속 conv 가 "무엇을 합칠지" 학습할 수 있다.
4. **Valid convolution + overlap-tile 전략**: 패딩 없는 conv 만 사용 → 출력은 입력의 "완전한 문맥이 있는" 중앙 영역. 큰 이미지는 타일을 겹치게 잘라 이어 붙여 처리하고, 경계 문맥은 **거울 반사(mirroring)** 로 채움 → GPU 메모리 한도 내에서 임의 크기 이미지를 매끈하게 처리.
5. **강한 데이터 증강**: 이동·회전·밝기 변화 + 특히 **탄성 변형(elastic deformation)**. 조직의 변형은 생체 영상의 가장 흔한 변동이라 효과적이며, 적은 주석 이미지로 학습 가능하게 한 핵심 요소.
6. **가중치 맵 손실**: 서로 붙어 있는 같은 클래스 객체(세포) 사이의 얇은 배경 경계를 정확히 분리하도록 해당 픽셀의 손실을 크게 가중.

## 3. 아키텍처/방법

### 3.1 네트워크 (원 논문 Figure 1, 입력 572×572)

| 단계 | 연산 | 특징맵 크기 / 채널 |
|---|---|---|
| 입력 | - | 572² × 1 |
| 인코더 1 | conv3×3 ×2 (+ReLU) → (skip 저장) → maxpool 2×2 | 568² × 64 → 284² |
| 인코더 2 | conv3×3 ×2 → maxpool | 280² × 128 → 140² |
| 인코더 3 | conv3×3 ×2 → maxpool | 136² × 256 → 68² |
| 인코더 4 | conv3×3 ×2 → maxpool | 64² × 512 → 32² |
| Bottleneck | conv3×3 ×2 (끝에 dropout) | 28² × 1024 |
| 디코더 4 | up-conv 2×2(채널 1024→512) → 인코더4 skip(중앙 crop 64²→56²) concat → conv3×3 ×2 | 56² × 512 → 52² |
| 디코더 3 | up-conv 2×2 → skip(136²→104²) concat → conv ×2 | 104² × 256 → 100² |
| 디코더 2 | up-conv → skip(280²→200²) concat → conv ×2 | 200² × 128 → 196² |
| 디코더 1 | up-conv → skip(568²→392²) concat → conv ×2 | 392² × 64 → 388² |
| 출력 | conv 1×1 | 388² × 2 (세포/배경) |

- 총 23개 합성곱 층, 약 **3,100만 파라미터**(구현 확인: 31.0M).
- 3×3 conv 는 패딩이 없어 매번 가로·세로가 2씩 줄어듦: 572 → 388 (테두리 92픽셀이 문맥 역할로만 쓰임).
- 풀링 직전의 특징맵 크기가 짝수가 되도록 입력 크기를 골라야 한다 (572 가 그 예).
- 초기화: 표준편차 $\sqrt{2/N}$ 의 가우시안 ($N$ = 한 뉴런의 입력 연결 수 = $k^2 C_{in}$), 곧 He 초기화. 깊은 망에서 층별 활성 크기가 일정하도록 하여 매우 중요하다고 논문이 강조.

### 3.2 손실: 가중치 맵이 있는 픽셀별 softmax 교차 엔트로피

$$p_k(\mathbf{x}) = \frac{\exp(a_k(\mathbf{x}))}{\sum_{k'}\exp(a_{k'}(\mathbf{x}))},\qquad E = \sum_{\mathbf{x}\in\Omega} w(\mathbf{x})\,\log p_{\ell(\mathbf{x})}(\mathbf{x})\;\;(\text{최소화할 때는 부호 반대})$$

$$w(\mathbf{x}) = w_c(\mathbf{x}) + w_0\cdot\exp\!\Big(-\frac{(d_1(\mathbf{x})+d_2(\mathbf{x}))^2}{2\sigma^2}\Big)$$

- $w_c$: 클래스 빈도 균형 가중치, $d_1, d_2$: 해당 픽셀에서 **가장 가까운 / 두 번째로 가까운 세포 경계까지의 거리**.
- 논문 설정: $w_0 = 10$, $\sigma \approx 5$ 픽셀. 두 세포 사이 좁은 틈에서는 $d_1+d_2$ 가 작아 가중치가 크게 올라간다 → 붙은 세포를 분리하는 "분리 배경 라벨"을 잘 맞추게 함.
- 가중치 맵은 정답 분할로부터 학습 전에 미리 계산.

### 3.3 학습 · 증강 · 추론

- Caffe, SGD. GPU 메모리를 효율적으로 쓰려고 **큰 입력 타일 + 배치 1**, 대신 momentum **0.99** 로 과거 샘플의 기울기를 크게 반영.
- 탄성 변형: 3×3 격자의 무작위 변위 벡터(가우시안, $\sigma = 10$ 픽셀)를 만들어 bicubic 보간으로 이미지·라벨에 동일 적용.
- 수축 경로 끝의 **dropout** 이 암묵적 데이터 증강 역할.
- 추론: 512×512 이미지 분할이 당시 GPU 에서 1초 이내. 큰 이미지는 overlap-tile.

## 4. 실험 결과 요약

(논문 보고값 기준)

| 과제 | 데이터 | 결과 |
|---|---|---|
| EM 신경 구조 분할 | ISBI 2012 (훈련 30장, 512×512) | 당시 1위. Ciresan 등의 슬라이딩 윈도우 CNN 을 warping error · Rand error 에서 능가, 추론은 훨씬 빠름. 후처리/앙상블 없이 단일 모델 |
| 광학 현미경 세포 분할 | ISBI 2015 Cell Tracking, PhC-U373 | mean IOU 약 0.92 (2위 약 0.83) |
| DIC 영상 세포 분할 | ISBI 2015 Cell Tracking, DIC-HeLa | mean IOU 약 0.78 (2위 약 0.46) |

- 훈련 데이터가 수십 장 수준뿐인데도 큰 격차로 1위 → 증강 + 구조의 효과.
- 학습 시간: 단일 GPU(당시 Titan 6GB)에서 약 10시간 수준으로 보고.

## 5. 의의 (앞뒤 논문과의 연결)

- **앞 (FCN)**: FCN 의 skip(합산 + 학습 업샘플) 아이디어를 **대칭 구조 + concat + 풍부한 디코더**로 확장. FCN 이 "분류망 → 분할망 변환" 이라면 U-Net 은 "분할을 위한 처음부터의 설계".
- **앞 (ResNet / DenseNet)**: skip 과 특징 재사용이라는 철학을 공유. concat 방식 skip 은 DenseNet 과 닮았고, 인코더 뒤 덧셈은 ResNet 계열.
- **앞 (FPN)**: top-down + lateral 구조는 U-Net 디코더와 개념적으로 같은 계보 (FPN 은 합산 + 각 레벨 예측, U-Net 은 concat + 최종 해상도 하나의 예측).
- **뒤 (Mask R-CNN)**: 인스턴스 분할로 가는 다른 길 — U-Net 은 픽셀 분류 + 후처리(가중치 맵/연결 성분)로 객체를 분리, Mask R-CNN 은 검출 후 RoI 별 마스크.
- **뒤 (DeepLab 계열 / Transformer 시대)**: U-Net 형태는 의료영상의 사실상 표준(3D U-Net, nnU-Net, UNet++, Attention U-Net, TransUNet, Swin-UNet)이며, 이후 **Diffusion 모델의 노이즈 예측망**(Stable Diffusion 의 UNet)에도 쓰이는 범용 구조가 되었다.

## 6. 구현 노트

`implementation/model.py`:
- `DoubleConv`(conv3×3-ReLU ×2), 인코더 4단 + bottleneck + 디코더 4단(`ConvTranspose2d(2×2, stride 2)`로 채널 절반) + 1×1 head.
- `padding=False`(기본)에서 논문 그대로 572 → 388 을 확인(`__main__`에서 assert), 중앙 crop 후 concat. 파라미터 수 31.0M 이 논문과 일치.
- `padding=True`(same) / `batch_norm=True` 는 현대적 변형 옵션. `output_size()` 는 valid 모드에서 가능한 입력 크기를 검증.
- He 초기화, dropout 옵션. 탄성 변형 증강과 overlap-tile 추론은 구현 범위 밖(요약 참고).

`implementation/train.py`: 합성 세포 데이터에 **논문의 가중치 맵($w_c$ + 경계 가중치)** 을 직접 구현해 가중 교차 엔트로피로 학습하고,
valid-conv 이므로 정답/가중치를 출력 크기로 중앙 crop 해 비교한다. foreground IoU 로 평가(합성 데이터라 논문 수치와 비교 불가).
