import torch
import torch.nn as nn

class ZBlock(nn.Module):
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

    def forward(self, x, y, z):
        """
        x: (B, Nx, d_model)
        y: (B, Ny, d_model)
        z: (B, K, d_model)
        """

        # z attends to x + y
        context = torch.cat([x, y], dim=1)

        out, _ = self.attn(z, context, context)
        z = self.norm1(z + out)

        out = self.ffn(z)
        z = self.norm2(z + out)

        return z

class YBlock(nn.Module):
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

    def forward(self, x, y, z):
        """
        x: (B, Nx, d_model)
        y: (B, Ny, d_model)
        z: (B, K, d_model)
        """

        # y attends to x + z
        context = torch.cat([x, z], dim=1)           # TODO : actual paper does only context=z type shi, no x again

        out, _ = self.attn(y, context, context)
        y = self.norm1(y + out)

        out = self.ffn(y)
        y = self.norm2(y + out)

        return y