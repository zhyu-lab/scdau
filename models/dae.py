# a diffusion autoencoder
# --------------------------------------------------------

import torch
import torch.nn as nn

from .ed import TriEncoder
from .unet import Linear_UNet


class DAE(nn.Module):
    """
    diffusion autoencoder
    """
    def __init__(
        self,
        input_dim=256,
        emb_size=64,
        hidden_dims=[512, 512, 256, 128]
    ):
        super().__init__()
        self.encoder = TriEncoder(input_dim=input_dim, emb_size=emb_size)
        self.unet_x = Linear_UNet(input_dim=input_dim, cond_dim=emb_size * 2, hidden_dims=hidden_dims)
        self.unet_y = Linear_UNet(input_dim=input_dim, cond_dim=emb_size * 2, hidden_dims=hidden_dims)

    def forward(self, x, y):
        z_x, z_y, z_x0, z_y0, z_x_a, z_y_a, z_xy= self.encoder(x, y)

        cond_x = torch.cat((z_x, z_x_a), dim=1)
        cond_y = torch.cat((z_y, z_y_a), dim=1)


        return z_x0, z_y0, cond_x, cond_y
