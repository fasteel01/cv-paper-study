# CV Paper Study

컴퓨터 비전(Computer Vision) 학부연구생 · 대학원 진학 준비를 위한 논문 스터디 저장소입니다.
매주 2편씩, **기초(고전 CNN) → 객체 탐지 → 분할(Segmentation) → Transformer/자기지도학습** 순서로
"밑바닥부터" 논문을 읽고, 핵심 내용을 정리하고, 모델을 직접 구현합니다.

- 매주 월요일 오전 8시(KST) 자동으로 새 논문 2편의 폴더가 생성되고, 요약 노트 + 구현 스타터 코드가 커밋됩니다.
  ([GitHub Actions 워크플로](./.github/workflows/weekly-paper.yml)로 GitHub 서버에서 실행되므로 로컬 컴퓨터가 꺼져 있어도 동작합니다. Actions 탭에서 수동 실행도 가능.)
- 로컬에서는 `git pull`로 최신 내용을 받아오면 됩니다. 원문 PDF는 저작권 문제로 저장소에 올리지 않고 로컬에만 보관합니다(`.gitignore`).
- 전체 커리큘럼과 진행 상태는 [`curriculum.yml`](./curriculum.yml)에서 관리합니다.

## 폴더 구조

```
papers/
  weekNN/
    <paper-slug>/
      summary.md            # 논문 요약 노트 (배경, 방법론, 핵심 수식, 실험 결과, 코멘트)
      paper_link.md          # 원문 PDF/arXiv 링크
      implementation/
        model.py             # 핵심 아키텍처를 처음부터(from scratch) 구현
        train.py              # (해당 시) 간단한 학습/검증 스크립트
        README.md             # 실행 방법
```

## 진행 상황

| Week | 논문 | 분류 | 상태 |
|---|---|---|---|
| 1 | [LeNet-5](./papers/week01/lenet5) — Gradient-Based Learning Applied to Document Recognition (1998) | foundations | ✅ |
| 1 | [AlexNet](./papers/week01/alexnet) — ImageNet Classification with Deep CNNs (2012) | foundations | ✅ |
| 2 | VGGNet — Very Deep Convolutional Networks (2014) | foundations | ⏳ |
| 2 | GoogLeNet — Going Deeper with Convolutions (2014) | foundations | ⏳ |
| 3 | BatchNorm (2015) | foundations | ⏳ |
| 3 | ResNet — Deep Residual Learning (2015) | foundations | ⏳ |
| 4 | DenseNet (2017) | foundations | ⏳ |
| 4 | MobileNet (2017) | foundations | ⏳ |
| 5 | R-CNN (2014) | detection | ⏳ |
| 5 | Fast R-CNN (2015) | detection | ⏳ |
| 6 | Faster R-CNN (2015) | detection | ⏳ |
| 6 | YOLOv1 (2016) | detection | ⏳ |
| 7 | SSD (2016) | detection | ⏳ |
| 7 | FPN (2017) | detection | ⏳ |
| 8 | RetinaNet (2017) | detection | ⏳ |
| 8 | FCN (2015) | segmentation | ⏳ |
| 9 | U-Net (2015) | segmentation | ⏳ |
| 9 | Mask R-CNN (2017) | segmentation | ⏳ |
| 10 | DeepLabv3 (2017) | segmentation | ⏳ |
| 10 | Transformer — Attention Is All You Need (2017) | transformer-era | ⏳ |
| 11 | ViT (2020) | transformer-era | ⏳ |
| 11 | DETR (2020) | transformer-era | ⏳ |
| 12 | Swin Transformer (2021) | transformer-era | ⏳ |
| 12 | CLIP (2021) | transformer-era | ⏳ |
| 13 | MAE (2021) | self-supervised | ⏳ |
| 13 | DINO (2021) | self-supervised | ⏳ |

## 참고

- 각 논문 폴더의 `paper_link.md`에 원문(arXiv 등) 링크가 있습니다. 원문 PDF는 해당 링크에서 직접 받아보실 수 있습니다.
- `summary.md`는 논문의 핵심을 정리한 노트이며, `implementation/`의 코드는 원 논문 아키텍처를 학습 목적으로 기초부터 재구현한 것입니다(공식 구현이 아닙니다).
