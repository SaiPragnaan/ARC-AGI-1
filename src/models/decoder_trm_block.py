import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + out)

        out = self.ffn(q)
        q = self.norm2(q + out)

        return q

class TRMBlock(nn.Module):
    def __init__(self, d_model, nhead, depth=4):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead)
            for _ in range(depth)
        ])

    def forward(self, query, context):
        """
        query: (B, *, d_model)
        context: (B, *, d_model)
        """

        for layer in self.layers:
            query = layer(query, context)

        return query
