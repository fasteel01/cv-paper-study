"""
Transformer (Vaswani et al., 2017) — from-scratch PyTorch implementation.

구성: 임베딩(+sinusoidal PE) -> Encoder x N -> Decoder x N (masked self-attn + cross-attn) -> Linear.
nn.MultiheadAttention / nn.Transformer 를 쓰지 않고 attention 을 직접 구현했다.

원 논문과의 차이점 / 단순화:
  - 기본 하이퍼파라미터는 데모용으로 작게 설정(d_model=128 등). 논문 base: d_model=512, h=8, d_ff=2048, N=6.
  - 논문은 Post-LN(LayerNorm(x + Sublayer(x))). `norm_first=True` 로 Pre-LN(현대적 관례, 학습 안정) 선택 가능.
  - BPE/word-piece 토크나이저는 포함하지 않고 정수 토큰 입력을 가정.
  - 마스크 규약: True = attention 허용, False = 차단.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """q,k,v: (..., T, d_k). mask: broadcastable to (..., Tq, Tk), True=허용."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.d_k = num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b = query.size(0)

        def split(x, w):  # (B, T, D) -> (B, h, T, d_k)
            return w(x).view(b, -1, self.h, self.d_k).transpose(1, 2)

        q, k, v = split(query, self.w_q), split(key, self.w_k), split(value, self.w_v)
        if mask is not None and mask.dim() == 3:  # (B, Tq, Tk) -> head 차원 추가
            mask = mask.unsqueeze(1)
        out, self.attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_first=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, src_mask):
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop(self.self_attn(h, h, h, src_mask))
            return x + self.drop(self.ffn(self.norm2(x)))
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_first=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, memory, tgt_mask, memory_mask):
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop(self.self_attn(h, h, h, tgt_mask))
            x = x + self.drop(self.cross_attn(self.norm2(x), memory, memory, memory_mask))
            return x + self.drop(self.ffn(self.norm3(x)))
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, memory_mask)))
        return self.norm3(x + self.drop(self.ffn(x)))


def subsequent_mask(size, device=None):
    """(1, T, T) lower-triangular, True=볼 수 있음 (미래 토큰 차단)."""
    return torch.tril(torch.ones(1, size, size, dtype=torch.bool, device=device))


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, num_heads=4, num_layers=3, d_ff=512,
                 dropout=0.1, max_len=512, pad_idx=0, norm_first=False, share_embeddings=False):
        super().__init__()
        self.d_model, self.pad_idx, self.norm_first = d_model, pad_idx, norm_first
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first) for _ in range(num_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first) for _ in range(num_layers))
        self.enc_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.dec_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.generator = nn.Linear(d_model, tgt_vocab, bias=False)
        if share_embeddings:  # 논문: 임베딩과 출력 projection 가중치 공유 (어휘가 같을 때)
            assert src_vocab == tgt_vocab
            self.src_emb.weight = self.tgt_emb.weight
            self.generator.weight = self.tgt_emb.weight
        for p in self.parameters():  # Xavier 초기화 (논문 구현 관례)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.src_emb.weight[pad_idx].zero_()
            self.tgt_emb.weight[pad_idx].zero_()

    def encode(self, src, src_mask):
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.dec_norm(x)

    def make_masks(self, src, tgt):
        src_mask = (src != self.pad_idx).unsqueeze(1)  # (B, 1, Ts) -> 브로드캐스트
        tgt_mask = (tgt != self.pad_idx).unsqueeze(1) & subsequent_mask(tgt.size(1), tgt.device)  # (B, Tt, Tt)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """src: (B, Ts), tgt: (B, Tt) (teacher forcing 입력) -> logits (B, Tt, tgt_vocab)"""
        src_mask, tgt_mask = self.make_masks(src, tgt)
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(tgt, memory, src_mask, tgt_mask))

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, max_len):
        src_mask = (src != self.pad_idx).unsqueeze(1)
        memory = self.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_mask = subsequent_mask(ys.size(1), src.device)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            nxt = self.generator(out[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        return ys[:, 1:]


if __name__ == "__main__":
    torch.manual_seed(0)
    for nf in (False, True):
        model = Transformer(src_vocab=100, tgt_vocab=100, norm_first=nf).eval()
        src = torch.randint(1, 100, (2, 10))
        src[1, 7:] = 0  # padding 예시
        tgt = torch.randint(1, 100, (2, 8))
        with torch.no_grad():
            logits = model(src, tgt)
        print(f"norm_first={nf}: src {tuple(src.shape)}, tgt {tuple(tgt.shape)} -> logits {tuple(logits.shape)}")
        assert logits.shape == (2, 8, 100)
    out = model.greedy_decode(src, bos_idx=1, max_len=5)
    print("greedy decode:", tuple(out.shape))

    # causal mask 검증: 미래 토큰을 바꿔도 이전 위치 출력은 불변이어야 함
    tgt2 = tgt.clone(); tgt2[:, -1] = (tgt2[:, -1] % 99) + 1
    with torch.no_grad():
        d = (model(src, tgt)[:, :-1] - model(src, tgt2)[:, :-1]).abs().max().item()
    print(f"causal check (max diff on earlier positions): {d:.2e}")
    assert d < 1e-5

    # 논문 base 설정 파라미터 수 (vocab 37000 공유 임베딩 기준 약 65M 근처)
    base = Transformer(37000, 37000, d_model=512, num_heads=8, num_layers=6, d_ff=2048, share_embeddings=True)
    print(f"base config params: {sum(p.numel() for p in base.parameters())/1e6:.1f}M")
