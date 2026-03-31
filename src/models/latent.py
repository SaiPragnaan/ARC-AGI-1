import torch
import torch.nn as nn

class LatentTokens(nn.Module):
    def __init__(self,num_tokens,d_model):
        super().__init__()
        self.z=nn.Parameter(torch.randn(num_tokens,d_model))
        self.cross_attn=nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )
    def forward(self,demo_features):
        # demo_features : (B, N, d_model) shi
        z=self.z.unsqueeze(0).expand(demo_features.size(0), -1, -1)
        z, _=self.cross_attn(z,demo_features,demo_features)

        return z

