# Week 7-1. SSD

**SSD: Single Shot MultiBox Detector**
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg — *ECCV*, 2016 (arXiv 1512.02325)
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

- **Faster R-CNN** 계열은 정확하지만 "영역 제안(RPN) → 특징 리샘플링(RoIPool) → 분류/회귀"의 2단계 구조라 느리다 (논문 기준 약 7 FPS).
- **YOLOv1**은 한 번의 forward로 검출해 빠르지만(약 45 FPS), 7×7 격자의 **단일 해상도** 특징 맵에서만 예측하므로 작은 객체와 군집한 객체에 약하고 정확도도 낮다.

SSD의 문제의식: **제안 단계와 특징 리샘플링 없이**, 한 번의 forward로 검출하되 정확도는 Faster R-CNN 수준으로 끌어올릴 수 없을까?
핵심 관찰은 *객체의 크기는 제각각이므로, 여러 해상도의 특징 맵에서 각각 예측하면 된다*는 것이다.

## 2. 핵심 아이디어

1. **멀티스케일 특징 맵에서의 예측**: 백본 뒤에 점점 작아지는 conv 층들을 덧붙이고, 해상도가 다른 6개 특징 맵 각각에서 직접 검출한다. 큰 특징 맵(해상도 높음)은 작은 객체, 작은 특징 맵은 큰 객체를 담당한다.
2. **Default box(앵커와 유사)**: 각 특징 맵 셀마다 여러 스케일·종횡비의 기준 박스를 깔고, 각 기준 박스에 대해 **클래스 점수 $c$개 + 박스 오프셋 4개**를 3×3 conv 하나로 예측한다.
3. **완전 합성곱 예측기**: 예측기가 3×3 conv라서 FC 층이 필요 없고 파라미터가 적다.
4. **Matching strategy + Hard Negative Mining**: 기준 박스와 GT를 IoU로 매칭하고, 압도적으로 많은 배경(negative) 박스는 손실이 큰 것만 골라 pos:neg = 1:3 으로 학습한다.
5. **강한 데이터 증강**: 랜덤 크롭(최소 IoU 조건), photometric 왜곡, 확대(zoom-out) 등이 특히 작은 객체 성능에 크게 기여한다 (논문 보고: VOC07에서 mAP 약 +8.8).

## 3. 아키텍처/방법

### 3.1 전체 구조 (SSD300)

백본은 **VGG-16**(ImageNet 사전학습)이며 분류용 FC 층을 합성곱으로 바꾼다. 입력은 300×300.

| 특징 맵 | 출처 | 해상도 | 채널 | 셀당 default box 수 | box 수 |
|---|---|---|---|---|---|
| ① | conv4_3 (L2 정규화) | 38×38 | 512 | 4 | 5776 |
| ② | fc7 (conv7) | 19×19 | 1024 | 6 | 2166 |
| ③ | conv8_2 | 10×10 | 512 | 6 | 600 |
| ④ | conv9_2 | 5×5 | 256 | 6 | 150 |
| ⑤ | conv10_2 | 3×3 | 256 | 4 | 36 |
| ⑥ | conv11_2 | 1×1 | 256 | 4 | 4 |
| 합계 | | | | | **8732** |

VGG-16 변형 (DeepLab-LargeFOV에서 차용):
- pool5를 2×2/2 → **3×3/1**(pad 1)로 바꿔 해상도를 유지.
- fc6, fc7을 각각 **dilated 3×3 conv(dilation 6, 1024채널)**, **1×1 conv(1024채널)** 로 변환 (원래 FC의 가중치를 subsampling해서 초기화).
- conv4_3은 앞쪽 층이라 활성값의 스케일이 커서 **L2 Normalization + 학습되는 채널별 스케일(초기값 20)** 을 적용 (ParseNet).
- 추가 층(extras): `1×1 conv(채널 축소) → 3×3 conv(stride 2, 또는 마지막 두 개는 valid)`를 반복해 10→5→3→1로 축소.

### 3.2 Default box 설계

$m$개의 특징 맵 중 $k$번째 맵의 default box 스케일:

$$s_k = s_{\min} + \frac{s_{\max} - s_{\min}}{m - 1}(k - 1), \quad k \in [1, m]$$

($s_{\min}=0.2$, $s_{\max}=0.9$.) 종횡비 $a_r \in \{1, 2, 3, \tfrac12, \tfrac13\}$ 에 대해

$$w_k^a = s_k \sqrt{a_r}, \qquad h_k^a = s_k / \sqrt{a_r}$$

그리고 $a_r = 1$ 일 때 $s'_k = \sqrt{s_k s_{k+1}}$ 인 박스를 하나 더 추가한다. 따라서 셀당 6개(또는 $a_r=3$ 계열을 뺀 4개)가 된다.
중심은 $\left(\frac{i+0.5}{|f_k|}, \frac{j+0.5}{|f_k|}\right)$ (정규화 좌표). 공식 Caffe 구현은 conv4_3에 0.1을 쓰고 나머지에 0.2 이후를 배치한다(크기 30, 60, 111, 162, 213, 264, 315 / 300) — 본 구현도 이 값을 따른다.

### 3.3 예측과 매칭

- 각 default box $d=(d_{cx}, d_{cy}, d_w, d_h)$ 마다 $c$개 클래스(배경 포함) 점수와 4개 오프셋을 예측.
- 박스 인코딩(GT $g$에 대한 회귀 타깃):

$$\hat g_{cx} = \frac{g_{cx} - d_{cx}}{d_w}, \quad \hat g_{cy} = \frac{g_{cy} - d_{cy}}{d_h}, \quad \hat g_w = \log\frac{g_w}{d_w}, \quad \hat g_h = \log\frac{g_h}{d_h}$$

  (공식 구현은 여기에 variance 0.1 / 0.2 를 나누어 타깃 스케일을 키운다.)
- **Matching**: (1) 각 GT에 대해 IoU가 가장 큰 default box를 매칭 (모든 GT가 최소 1개 매칭되도록), (2) 나머지 default box 중 어떤 GT와 IoU > 0.5 이면 그 GT에 매칭. 한 GT에 **여러 default box** 가 매칭되는 것을 허용 → 학습이 쉬워진다.

### 3.4 손실 함수

$$L(x, c, l, g) = \frac{1}{N}\bigl(L_{\text{conf}}(x, c) + \alpha L_{\text{loc}}(x, l, g)\bigr), \quad \alpha = 1$$

- $N$: 매칭된 default box 수 ($N=0$이면 손실 0).
- $L_{\text{loc}}$: positive 박스에 대한 **Smooth L1** 손실.
- $L_{\text{conf}}$: softmax cross-entropy. positive는 매칭된 GT의 클래스, negative는 배경(0).
- **Hard negative mining**: negative를 배경 신뢰도 손실이 큰 순으로 정렬해 상위만 선택, **pos : neg ≈ 1 : 3**. (안 하면 배경이 압도해 학습이 느리고 불안정.)

### 3.5 추론

모든 8732개 박스를 디코딩 → 클래스별 confidence threshold(예: 0.01) → **클래스별 NMS**(IoU 0.45) → 이미지당 상위 200개 정도 유지.

### 3.6 학습 세부

SGD(momentum 0.9, weight decay 5e-4), 초기 lr은 $10^{-3}$ 정도에서 시작해 단계적으로 감소. 데이터 증강: 최소 IoU 조건 크롭/랜덤 크롭 + 수평 반전 + 색 왜곡 + zoom-out.

## 4. 실험 결과 요약

(수치는 논문 보고값 기준, 약 값)

| 모델 | 데이터 | mAP | 속도 |
|---|---|---|---|
| Faster R-CNN (VGG16) | VOC07 test | 약 73.2 | 약 7 FPS |
| YOLOv1 | VOC07 test | 약 63.4 | 약 45 FPS |
| **SSD300** | VOC07 test | **약 74.3** | **약 46 FPS** |
| **SSD512** | VOC07 test | 약 76.8 | 약 19 FPS |

- SSD300 이 속도와 정확도 모두에서 Faster R-CNN, YOLOv1 을 능가한다. (07+12 학습 시 SSD300 약 79.4, SSD512 약 82.2 mAP.)
- COCO test-dev: SSD300 은 mAP@[.5:.95] 약 23, SSD512 약 27 수준.
- **Ablation**: 데이터 증강 제거 시 큰 폭 하락, 다양한 종횡비의 default box와 atrous(dilated) conv 사용이 도움, 여러 층에서 예측할수록 mAP 상승.
- **약점**: 작은 객체에 상대적으로 약함 — 얕은 층(conv4_3)은 의미 정보가 약하기 때문. 이는 다음 논문 FPN의 직접적 동기가 된다.

## 5. 의의 (앞뒤 논문과의 연결)

- **앞**: Faster R-CNN의 앵커(anchor)를 **여러 해상도 특징 맵으로 확장**하고 RoI 단계 없이 분류+회귀를 한 번에 수행한 것이 SSD. YOLOv1의 "단일 forward" 철학에 Faster R-CNN의 "앵커 기반 회귀"를 합친 형태다.
- **뒤**: 얕은 층의 의미 약함 문제는 **FPN**(top-down 경로로 고수준 의미를 전달)이 해결한다. 또한 dense 검출기의 극단적 클래스 불균형(hard negative mining으로 우회)은 **RetinaNet의 Focal Loss**가 근본적으로 해결한다.
- 원스테이지 검출기의 표준 설계(멀티스케일 head, default box 매칭, NMS)를 정립했다.

## 6. 구현 노트

`implementation/model.py` 에 다음을 처음부터 구현했다.
- VGG-16 base(dilated fc6, 1×1 fc7), L2Norm, extras, 6개 분류/회귀 head → `SSD300`
- `generate_default_boxes()`: 총 8732개 default box 생성
- `encode/decode`, `box_iou`, `match()`(양방향 매칭), `MultiBoxLoss`(Smooth L1 + hard negative mining 3:1), 클래스별 NMS 후처리 `detect()`

원 논문과의 차이: ImageNet 사전학습 가중치 로딩, fc6/fc7 subsampling 초기화, 데이터 증강은 포함하지 않았다.
`train.py` 는 데이터셋 없이 더미 데이터로 손실 계산과 학습 step을 보여준다.
