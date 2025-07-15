from utils.attention_ops import repeat_kv
from utils.attention_ops import apply_rotary_pos_emb
from utils.attention_ops import build_padding_mask
from config import MiniBertConfig
import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,config:MiniBertConfig):
        super(Attention, self).__init__()
        self.hidden_size = config.hidden_size
        self.head_num_q = config.num_attention_heads
        self.head_num_kv = config.num_key_value_heads
        self.head_dim = config.hidden_size//self.head_num_q

        assert self.hidden_size % self.head_num_q == 0
        assert self.head_num_q % self.head_num_kv== 0

        self.q = nn.Linear(config.hidden_size, self.head_num_q * self.head_dim)
        self.k = nn.Linear(config.hidden_size, self.head_num_kv * self.head_dim)
        self.v = nn.Linear(config.hidden_size, self.head_num_kv * self.head_dim)
        self.o = nn.Linear(self.head_num_q * self.head_dim, config.hidden_size)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self,x,use_cache,position_embeddings,past_kv,attention_mask):
        b,s,d = x.shape
        q = self.q(x).view(b,s,self.head_num_q,self.head_dim).transpose(1, 2)
        k = self.k(x).view(b,s,self.head_num_kv,self.head_dim).transpose(1, 2)
        v = self.v(x).view(b,s,self.head_num_kv,self.head_dim).transpose(1, 2)

        # 此处默认为None（采用bert embedding）
        if position_embeddings is not None:
            q, k = apply_rotary_pos_emb(q, k, position_embeddings)

            # === KV Cache 拼接 ===
        if past_kv is not None:
            past_k, past_v = past_kv  # [B, H_kv, S_past, D]
            k = torch.cat([past_k, k], dim=2)  # 拼接到时间维
            v = torch.cat([past_v, v], dim=2)
        present_kv =(k,v)

            # === Repeat KV（KV头扩展） ===
        k = repeat_kv(k, self.head_num_q // self.head_num_kv)
        v = repeat_kv(v, self.head_num_q // self.head_num_kv)

        # flash的attention实现:后补

        # 普通的attention实现
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores += build_padding_mask(attention_mask)
        probs = F.softmax(scores,dim=-1)
        probs = self.attn_dropout(probs)
        output = torch.matmul(probs,v)
        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.o(output)
        output = self.out_dropout(output)
        return output,present_kv

