# DeepLabv3 구현

`model.py`: Atrous ResNet(output stride 8/16/32, multi-grid) + ASPP(1×1, atrous×3, image pooling) + 분류 head. 사전학습 가중치 없음.
`train.py`: 더미 데이터로 cross-entropy(ignore_index=255), poly lr, 몇 step 학습을 시연 (PASCAL VOC 는 다운로드하지 않음).

## 실행 방법

```bash
pip install torch
python model.py            # OS=32/16/8 에서 feature/logit shape 확인
python train.py --steps 5  # 더미 데이터 학습 데모
```
