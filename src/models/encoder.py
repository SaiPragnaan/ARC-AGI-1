import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,d_model, nhead, num_layers):
        super().__init__()

        self.d_model=d_model
        self.nhead=nhead
        self.num_layers=num_layers
        
        layer=nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    batch_first=True,
                    dim_feedforward=4*d_model,
                    activation="gelu"
                )
        self.encoder=nn.TransformerEncoder(
                    encoder_layer=layer,
                    num_layers=num_layers
                )

    def forward(self, x):
        return self.encoder(x)   # x : (N, d_model) shi