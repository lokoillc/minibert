from typing import Tuple
import torch
from torch import nn

# 1.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1) # q, k: [B, H_kv, T, D]

# 2.build_rope and apply_rope
def build_rope_cache(seq_len, head_dim, device, dtype=torch.float32, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    idx_theta = torch.outer(seq_idx, theta)  # [seq_len, dim/2]
    # [seq_len, dim]
    sin, cos = torch.sin(idx_theta), torch.cos(idx_theta)
    cos = torch.stack([cos, cos], dim=-1).reshape(1, 1, seq_len, head_dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(1, 1, seq_len, head_dim)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, T, H, D]
    # cos, sin: [T, 1, D]
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)
    q_embed = (q * cos) + (rotate(q) * sin)
    k_embed = (k * cos) + (rotate(k) * sin)
    return q_embed, k_embed

# 3.build_pe and apply_pe
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        input_ids: [B, S]
        """
        b, s = input_ids.shape
        position_ids = torch.arange(s, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # [1, S]
        token_emb = self.token_embeddings(input_ids)       # [B, S, D]
        pos_emb = self.position_embeddings(position_ids)   # [1, S, D]
        x = token_emb + pos_emb
        return self.dropout(self.layer_norm(x))

# 4.padding_mask
def build_padding_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    构造 padding mask，仅遮盖 padding token（适用于 BERT）

    参数：
        attention_mask: [B, S] → 1 表示有效，0 表示 padding
        dtype: 输出类型（如 float32）

    返回：
        padding_mask: [B, 1, 1, S]，可直接用于 scores += mask
    """
    return (1.0 - attention_mask[:, None, None, :]).to(dtype=dtype) * -1e9
