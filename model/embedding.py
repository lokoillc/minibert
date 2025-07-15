import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)  # type 0 for A, 1 for B
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        b, s = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 构造 position_ids: shape = [1, s]
        position_ids = torch.arange(s, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(b, -1)

        token_emb = self.token_embeddings(input_ids)        # [B, S, D]
        pos_emb = self.position_embeddings(position_ids)    # [B, S, D]
        seg_emb = self.segment_embeddings(token_type_ids)   # [B, S, D]

        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
