import torch
import torch.nn as nn

class GridEmbedding(nn.Module):
    def __init__(self, d_model,max_size=30,max_pairs=12):
        super().__init__()

        self.color_embed=nn.Embedding(10,d_model)  ## TODO : VERIFY WITH LESS THAB d_model like 32 - 64 type shi
        self.row_embed = nn.Embedding(max_size,d_model)
        self.col_embed = nn.Embedding(max_size,d_model)
        self.pair_embed =nn.Embedding(max_pairs+1,d_model)   # 0 reserved for test io, so +1 doing 
        self.io_embed=nn.Embedding(2, d_model)
        
    def forward(self, grid, pair_id=0, is_output=0):
        H, W = grid.shape
        device = grid.device

        rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W)

        x = self.color_embed(grid)
        x = x + self.row_embed(rows) + self.col_embed(cols)

        # pair embedding
        pair = torch.full((H, W), pair_id, device=device)
        x = x + self.pair_embed(pair)

        # IO embedding
        io = torch.full((H, W), is_output, device=device)
        x = x + self.io_embed(io)

        return x