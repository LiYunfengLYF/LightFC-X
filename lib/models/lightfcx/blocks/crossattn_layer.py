import torch.nn as nn
from .spatial_crossattn import crossattn_spatial, crossattn_spatial_relu
from lib.utils.token_utils import token2patch, patch2token

class crosspath_cat2add_linear2conv1_croattn_spatial(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_attn = crossattn_spatial(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.end_proj2 = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        v1, v2 = self.cross_attn(x1, x2)
        y1 = x1 + v1
        y2 = x2 + v2
        out_x1 = self.norm1(x1 + patch2token(self.end_proj1(token2patch(y1))))
        out_x2 = self.norm2(x2 + patch2token(self.end_proj2(token2patch(y2))))
        return out_x1, out_x2


class crosspath_cat2add_linear2conv1_croattn_spatial_relu(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_attn = crossattn_spatial_relu(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.end_proj2 = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        v1, v2 = self.cross_attn(x1, x2)
        y1 = x1 + v1
        y2 = x2 + v2
        out_x1 = self.norm1(x1 + patch2token(self.end_proj1(token2patch(y1))))
        out_x2 = self.norm2(x2 + patch2token(self.end_proj2(token2patch(y2))))
        return out_x1, out_x2