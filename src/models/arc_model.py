import torch
import torch.nn as nn

from embedding import GridEmbedding
from encoder import Encoder
from latent import LatentTokens
from decoder_trm_block import ZBlock, YBlock

class ARCModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config["d_model"]

        self.embed = GridEmbedding(d_model)

        self.encoder = Encoder(d_model, nhead=4, num_layers=3)

        self.latent = LatentTokens(num_tokens=config["K"], d_model=d_model)

        self.z_block = ZBlock(d_model, nhead=4)
        self.y_block = YBlock(d_model, nhead=4)

        self.output_head = nn.Linear(d_model, 10)

        self.N_sup = config["N_sup"]
        self.n_z = config["n_z"]

    def forward(self, demos, test_input):

        B = 1  # for now (batching later)

        demo_feats = []

        for i, (x, y) in enumerate(demos):
            x = self.embed(x, pair_id=i+1, is_output=0)
            y = self.embed(y, pair_id=i+1, is_output=1)

            xy = torch.cat([x, y], dim=0)
            xy = xy.view(-1, xy.shape[-1])

            h = self.encoder(xy.unsqueeze(0))
            demo_feats.append(h)

        demo_feats = torch.cat(demo_feats, dim=1)  # (1, N, d_model)

        z = self.latent(demo_feats)

        x = self.embed(test_input, pair_id=0, is_output=0)
        x = x.view(1, -1, z.shape[-1])

        y = torch.zeros_like(x)

        outputs = []

        for _ in range(self.N_sup):

            for _ in range(self.n_z):
                z = self.z_block(x, y, z)

            y = self.y_block(x, y, z)

            logits = self.output_head(y)
            outputs.append(logits)

        return outputs