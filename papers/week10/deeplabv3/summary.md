# Week 10. DeepLabv3

**Rethinking Atrous Convolution for Semantic Image Segmentation**
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam — *arXiv*, 2017
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

FCN(Week 8) 이후 시맨틱 분할은 분류용 CNN을 fully convolutional 로 바꿔 쓰는 방식이 표준이 되었다. 그런데 분류용 CNN은
연속적인 stride/pooling 으로 해상도를 크게 줄이기 때문에(보통 output stride 32) 분할에 필요한 세밀한 위치 정보가 손실된다.
DeepLab 계열은 두 가지 난점을 다룬다.

1. **해상도 감소**: 반복되는 pooling/stride 로 feature map 이 작아짐 → **atrous(dilated) convolution** 으로 해상도를 유지한 채 receptive field 를 확장.
2. **다중 스케일 객체**: 같은 클래스라도 이미지 속 크기가 크게 다름 → 여러 스케일의 문맥을 함께 포착해야 함.

DeepLabv1/v2 는 후처리로 DenseCRF 를, 다중 스케일은 ASPP(Atrous Spatial Pyramid Pooling)로 다뤘다. v3 는 atrous convolution 을
"다시 생각"하여 **CRF 후처리 없이** 모듈 설계만으로 성능을 끌어올린다.

## 2. 핵심 아이디어

1. **Atrous convolution 을 backbone 뒤쪽 블록에 cascade 로 적용**: 마지막 몇 개 블록의 stride 를 제거하고 대신 dilation rate 를 키워
   output stride(OS)를 8 또는 16 으로 유지. 연속된 블록을 복제하여 깊게 쌓은 구조(cascade)와도 비교.
2. **Multi-grid**: 한 블록 안의 세 conv 에 서로 다른 dilation 배수 (예: (1,2,4))를 곱해 더 넓은 범위의 문맥을 확보.
3. **ASPP 개선**: 병렬 분기로 1×1 conv, 3×3 atrous conv 3개(OS=16에서 rate 6, 12, 18), 그리고 **image-level feature(global average pooling)** 를 결합.
   모든 분기에 **BatchNorm** 사용. 큰 rate 일수록 feature map 가장자리에서 유효한 가중치가 줄어 3×3 이 사실상 1×1 로 퇴화하는 문제를
   image pooling 분기가 보완.
4. **DenseCRF 제거**: 후처리 없이도 v2 보다 우수.

## 3. 아키텍처/방법

### Atrous convolution

1D 신호 기준 출력 `y[i] = Σ_k x[i + r·k] · w[k]` (r = atrous rate). r=1 이면 일반 conv.
파라미터 수와 연산 형태는 그대로이면서 유효 커널 크기는

$$k_{eff} = k + (k-1)(r-1)$$

즉 3×3, r=6 이면 유효 13×13. 공간 해상도를 줄이지 않고 receptive field 를 넓힌다.

### Output stride 와 dilation

ResNet 은 conv1(÷2) · pool(÷2) · layer2(÷2) · layer3(÷2) · layer4(÷2) 로 OS=32. DeepLabv3 는 layer4 (OS=16) 또는 layer3, 4 (OS=8)의
stride 를 1 로 바꾸고 이후 conv 의 dilation 을 그만큼 곱한다.

| OS | layer3 stride/dilation | layer4 stride/dilation | ASPP rates |
|---|---|---|---|
| 32 | 2 / 1 | 2 / 1 | (분류 원형) |
| 16 | 2 / 1 | 1 / 2 | 6, 12, 18 |
| 8 | 1 / 2 | 1 / 4 | 12, 24, 36 |

Multi-grid (예 (1,2,4))를 쓰면 layer4 블록 내 3개 residual unit 의 dilation 은 `base_rate × grid` 가 된다(예: OS=16, base 2 → 2, 4, 8).

### ASPP (v3)

```
feature (2048, H/OS, W/OS)
 ├─ 1×1 conv (256) + BN + ReLU
 ├─ 3×3 conv, rate r1 (256) + BN + ReLU
 ├─ 3×3 conv, rate r2 (256) + BN + ReLU
 ├─ 3×3 conv, rate r3 (256) + BN + ReLU
 └─ global avg pool → 1×1 conv (256) + BN + ReLU → bilinear upsample
concat (1280) → 1×1 conv (256) + BN + ReLU (+ dropout) → 1×1 conv (num_classes) → bilinear upsample ×OS
```

### 학습 세부

- Loss: 픽셀별 cross-entropy. 정답을 다운샘플하는 대신 **로짓을 입력 해상도로 업샘플**하여 비교(세밀한 GT 유지).
- Learning rate: "poly" 스케줄 `lr = base · (1 − iter/max_iter)^0.9`.
- 증강: 랜덤 스케일(0.5~2.0), 크롭, 좌우 반전. 큰 crop(513×513)과 batch size 16 에서 BN 통계를 함께 학습.
- 추론: OS=8 + 멀티스케일/플립 입력으로 성능을 더 끌어올림.

## 4. 실험 결과 요약

(수치는 논문 기억 기반의 근사치)

- ResNet-101 기반 PASCAL VOC 2012 val 에서 OS 를 32 → 16 → 8 로 줄이면 mIoU 가 꾸준히 상승. cascade 보다 ASPP 병렬 구조가 우수.
- Multi-grid 와 image pooling 의 추가가 각각 성능을 개선하며, 최종 val mIoU 는 80% 안팎(MS-COCO 사전학습, 멀티스케일+플립 사용 시).
- VOC 2012 test 에서 약 85.7% mIoU 로 당시 최고 수준, CRF 없이 달성. Cityscapes 에서도 경쟁력 있는 결과를 보고.

## 5. 의의 (앞뒤 논문과의 연결)

- **FCN(Week 8)** 이 "분류망 → dense prediction" 을 열었다면, DeepLabv3 는 *해상도를 잃지 않고* 문맥을 얻는 방법(atrous)을 체계화.
- **U-Net(Week 9)** 은 skip connection 인코더-디코더로 해상도를 복원하지만, DeepLabv3 는 dilation 으로 애초에 해상도를 유지. (이후 DeepLabv3+ 는 디코더를 추가해 둘을 절충.)
- **ResNet(Week 3)** backbone 을 수정해 사용하며, **BatchNorm** 이 ASPP 전체에 적용됨.
- ASPP 의 멀티스케일 문맥 모델링 아이디어는 이후 PSPNet 등과 함께 분할의 표준이 되었고, 다음 주 **Transformer** 기반 모델(ViT, Swin)이 전역 문맥을 어텐션으로 얻는 흐름과 대비해 볼 수 있다.

## 6. 구현 노트

`implementation/model.py`:
- ResNet-50 스타일 bottleneck backbone 을 직접 구현하고 `output_stride`(8/16/32)에 따라 stride 를 dilation 으로 교체. (사전학습 가중치는 사용하지 않음)
- `multi_grid` 옵션, ASPP(1×1 + 3 atrous + image pooling), 분류 head, 로짓 bilinear 업샘플까지 구현.
- 원 논문은 ResNet-101 을 쓰지만 단순화를 위해 기본 ResNet-50 구성(블록 수 (3,4,6,3)) 사용. 블록 수 인자로 변경 가능.
- 학습 시 BN 은 작은 배치에서 불안정하므로 데모에서는 batch 2 + train 모드이며, 실제 재현에는 큰 배치/ SyncBN 이 필요.

`implementation/train.py`: PASCAL VOC 는 용량이 커서 더미 데이터로 loss 계산, poly lr, 1~수 step 학습만 시연.
