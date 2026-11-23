# Week 10. Transformer

**Attention Is All You Need**
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin — *NeurIPS*, 2017
원문: [`paper_link.md`](./paper_link.md) 참고

## 1. 배경 및 문제의식

당시 시퀀스 변환(기계번역 등)의 표준은 RNN/LSTM/GRU 기반 encoder-decoder(+ attention)였다. 이 구조는 다음 한계가 있었다.

- **순차 계산**: 시점 t 의 은닉 상태가 t−1 에 의존 → 시퀀스 길이 방향 병렬화가 불가능하여 학습이 느림.
- **장거리 의존성**: 멀리 떨어진 토큰 사이 정보가 O(n) 단계를 거쳐야 전달되어 학습이 어려움.

CNN 기반 모델(ByteNet, ConvS2S)은 병렬화는 되지만 두 위치를 연결하는 연산 수가 거리에 따라 증가(선형/로그)한다.
Transformer 는 **recurrence 와 convolution 을 완전히 제거하고 attention 만으로** 시퀀스를 모델링하여
병렬화와 임의 두 위치 간 O(1) 경로 길이를 동시에 달성한다.

## 2. 핵심 아이디어

1. **Self-attention**: 시퀀스 내 모든 위치쌍이 직접 상호작용.
2. **Scaled dot-product attention**: `softmax(QKᵀ/√d_k)·V`. √d_k 로 나누어 큰 d_k 에서 softmax 포화(그래디언트 소실)를 방지.
3. **Multi-head attention**: 여러 부분공간에서 병렬로 attention 을 수행하여 서로 다른 관계 유형을 포착.
4. **Positional encoding**: 순서 정보가 없는 attention 에 사인/코사인 위치 인코딩을 더함.
5. **Residual + LayerNorm, Position-wise FFN**: 깊게 쌓아도 학습 가능하도록 하는 구조 요소.
6. **Masked self-attention(decoder)**: 미래 토큰을 보지 못하게 하여 auto-regressive 성질 유지.

## 3. 아키텍처/방법

### Scaled dot-product attention

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### Multi-head attention

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad \mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$

base 모델: h=8, d_k=d_v=d_model/h=64. 헤드별 차원을 줄여 총 연산량은 단일 헤드(전체 차원)와 비슷.

### Position-wise Feed-Forward

$$\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$$

d_model=512 → d_ff=2048 → 512.

### Positional Encoding

$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d_{model}}\big),\quad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d_{model}}\big)$$

임의의 오프셋 k 에 대해 PE(pos+k)가 PE(pos)의 선형변환으로 표현되어 상대 위치 학습이 쉬울 것이라는 가설. (학습형 임베딩과 성능이 거의 같다고 보고.)

### 전체 구조

| 구성 | Base | Big |
|---|---|---|
| 인코더/디코더 레이어 수 N | 6 | 6 |
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| 헤드 수 h | 8 | 16 |
| dropout | 0.1 | 0.3(EN-DE 기준 약 0.3, EN-FR 은 0.1) |
| 파라미터 | 약 65M | 약 213M |

- **Encoder layer**: Self-Attention → Add&Norm → FFN → Add&Norm
- **Decoder layer**: Masked Self-Attention → Add&Norm → Cross-Attention(Q: decoder, K,V: encoder 출력) → Add&Norm → FFN → Add&Norm
- 입력/출력 임베딩과 출력 softmax 직전 선형층의 **가중치 공유**, 임베딩에 √d_model 곱함.
- Post-LN: `LayerNorm(x + Sublayer(x))` (원 논문 방식. 이후 Pre-LN 이 학습 안정성 때문에 흔히 사용됨.)

### 학습 세부

- Optimizer: Adam(β1=0.9, β2=0.98, ε=1e-9)
- LR 스케줄(warmup): `lrate = d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)`, warmup=4000
- 정규화: residual dropout 0.1, **label smoothing** ε=0.1 (perplexity 는 나빠지나 BLEU/정확도 향상)
- 데이터: WMT 2014 EN-DE(약 450만 문장쌍), EN-FR(약 3600만), BPE/word-piece 사용.

### 복잡도 비교 (논문 Table 1, n=시퀀스 길이, d=표현 차원)

| 레이어 | 레이어당 복잡도 | 순차 연산 | 최대 경로 길이 |
|---|---|---|---|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |

n < d 인 경우(문장 번역)에 self-attention 이 더 효율적. 이미지처럼 n 이 매우 크면 O(n²) 이 병목이 된다(이후 ViT/Swin 의 동기).

## 4. 실험 결과 요약

(수치는 논문 기억 기반 근사치)

- WMT14 EN→DE: Big 모델 BLEU 약 28.4 로 당시 최고(앙상블 포함 기존 결과를 2 BLEU 이상 상회).
- WMT14 EN→FR: Big 모델 BLEU 약 41.8 (단일 모델 SOTA), 학습 비용은 기존 대비 훨씬 적음(8 P100 GPU 에서 약 3.5일).
- Base 모델도 이전 SOTA 앙상블을 능가하는 성능을 훨씬 적은 학습 비용으로 달성.
- Ablation: 헤드 수는 너무 적거나(1) 너무 많아도 성능 하락, d_k 를 줄이면 성능 하락, 큰 모델·dropout 이 유익, 학습형 PE 와 사인 PE 는 거의 동등.
- 영어 constituency parsing 에도 일반화되어 전이 가능함을 보임.

## 5. 의의 (앞뒤 논문과의 연결)

- NLP 의 BERT, GPT 계열의 기반이 되었고, 이 구조가 비전으로 확장된 것이 **ViT(Week 11)**, 탐지에 적용한 것이 **DETR(Week 11)**, 계층적 윈도우 attention 이 **Swin(Week 12)**.
- CNN(ResNet 등)이 가진 locality/translation-equivariance 의 inductive bias 없이 데이터로부터 관계를 학습 → 대규모 데이터에서 강력.
- 이번 주 DeepLabv3 가 dilation/ASPP 로 문맥을 확장했다면, Transformer 는 모든 위치 쌍 간 **전역 문맥을 한 레이어에서 직접** 얻는다는 점이 대비된다.
- 한계: 시퀀스 길이에 대한 O(n²) 메모리/연산 → 이후 효율적 attention 연구의 동기.

## 6. 구현 노트

`implementation/model.py`:
- `scaled_dot_product_attention`, `MultiHeadAttention`, `PositionwiseFeedForward`, `PositionalEncoding`, `EncoderLayer`, `DecoderLayer`, `Transformer`를 `nn.MultiheadAttention`/`nn.Transformer` 없이 직접 구현.
- 마스크: padding mask 와 causal(subsequent) mask 를 bool 로 구성(True=attend 가능).
- 단순화: 기본값은 base 보다 작은 설정으로 데모. 원 논문 Post-LN 을 따르되 `norm_first=True` 로 Pre-LN 선택 가능.
  임베딩-출력 가중치 공유는 `share_embeddings`로 옵션 제공(소스/타깃 어휘가 같을 때).
- 논문의 warmup LR 스케줄(`noam_lr`)과 label smoothing 손실은 `train.py` 에서 사용.

`implementation/train.py`: 외부 데이터 없이 합성 **sequence reversal(시퀀스 뒤집기)** 과제로 학습·정확도를 확인하고,
greedy decoding 으로 추론 과정을 보여준다.
