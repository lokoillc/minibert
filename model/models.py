import torch
import torch.nn as nn
from model.embedding import BertEmbeddings
from config import MiniBertConfig
from model.block import MiniBertBlock
from model.norm import RMSNorm

class MiniBertModel(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            dropout=config.dropout,
        )

        self.layers = nn.ModuleList([
            MiniBertBlock(
                config,
                use_moe=config.use_moe,
                moe_num_experts=config.moe_num_experts,
                moe_top_k=config.moe_top_k
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                use_cache=False, position_embeddings=None, past_kvs=None):
        """
        input_ids: [B, S]
        token_type_ids: [B, S]
        attention_mask: [B, S]
        position_embeddings: 可选的位置编码（如 RoPE 的坐标）
        past_kvs: List[(past_k, past_v), ...] for KV Cache
        """

        hidden_states = self.embeddings(input_ids, token_type_ids)

        load_loss_total = 0.0
        presents = []  # for KV cache output

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None

            hidden_states, load_loss, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                past_kv=past_kv
            )

            if use_cache:
                presents.append(present_kv)

            if load_loss is not None:
                load_loss_total = load_loss_total + load_loss

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,  # [B, S, D]
            "load_loss": load_loss_total,
            "past_kvs": presents if use_cache else None# 如有支持 KV Cache 输出再开启
        }





