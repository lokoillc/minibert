import torch
from torch import nn
from config import MiniBertConfig
from typing import Tuple
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super(FFN, self).__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
    def forward(self,x):
        return self.dropout(self.down(self.activation(self.gate(x))*self.up(x)))

class MOEFFN(nn.Module):
    def __init__(self, config: MiniBertConfig, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控
        self.gate = nn.Linear(self.hidden_size, self.num_experts)

        # experts（共享FFN架构，但各自独立参数）
        self.experts = nn.ModuleList([FFN(config) for _ in range(self.num_experts)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        x: [B, S, D]
        return:
            output: [B, S, D]
            load_balance_loss: scalar
        """
        B, S, D = x.shape
        x_flat = x.view(B * S, D)  # [BS, D]

        # === 1. 门控 ===
        gate_logits = self.gate(x_flat)  # [BS, E]
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # [BS, top_k]

        # === 2. 计算 softmax 权重 ===
        topk_weights = F.softmax(topk_vals, dim=-1)  # [BS, top_k]

        # === 3. 准备所有 experts 计算 ===
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat))  # 每个: [BS, D]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [BS, E, D]

        # === 4. 获取topk expert outputs ===
        # topk_idx: [BS, top_k] -> gather
        topk_exp_out = torch.gather(expert_outputs, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))  # [BS, top_k, D]

        # === 5. 按topk softmax权重加权 ===
        output = (topk_exp_out * topk_weights.unsqueeze(-1)).sum(dim=1)  # [BS, D]
        output = output.view(B, S, D)  # 还原为 [B, S, D]
        output = self.dropout(output)

        # === 6. 负载均衡损失 ===
        probs = F.softmax(gate_logits, dim=-1)  # [BS, E]
        load = probs.mean(dim=0)  # [E]
        load_balance_loss = (load * self.num_experts).pow(2).mean()

        return output, load_balance_loss
