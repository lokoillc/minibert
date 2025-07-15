import torch.nn as nn
from model.attention import Attention
from model.moeffn import FFN, MOEFFN
from model.norm import RMSNorm
from config import MiniBertConfig

class MiniBertBlock(nn.Module):
    def __init__(self, config: MiniBertConfig, use_moe: bool = False, moe_num_experts: int = 4, moe_top_k: int = 1):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config=config)
        self.dropout1 = nn.Dropout(config.dropout)

        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout)

        self.use_moe = use_moe
        if use_moe:
            self.ffn = MOEFFN(config, num_experts=moe_num_experts, top_k=moe_top_k)
        else:
            self.ffn = FFN(config)

    def forward(self, x, attention_mask, use_cache=False, position_embeddings=None, past_kv=None):
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(
            x_norm,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            past_kv=past_kv,
            attention_mask=attention_mask,
        )
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        if self.use_moe:
            ffn_out, load_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            load_loss = None

        x = x + self.dropout2(ffn_out)
        return x, load_loss, present_kv

