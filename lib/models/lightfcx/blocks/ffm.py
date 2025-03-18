import torch
import torch.nn as nn

from .crossattn_layer import crosspath_cat2add_linear2conv1_croattn_spatial, \
    crosspath_cat2add_linear2conv1_croattn_spatial_relu
from .joint_feat_encoding import jfe


class ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = crosspath_cat2add_linear2conv1_croattn_spatial(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = jfe(in_channels=dim * 2,
                               out_channels=dim,
                               reduction=reduction,
                               norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge



class ffm_crospath_cat2add_linear2conv1_croattn_spatial_relu_ffn_357_stage2_unshare(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = crosspath_cat2add_linear2conv1_croattn_spatial_relu(dim=dim, reduction=reduction,
                                                                         num_heads=num_heads)
        self.channel_emb1 = jfe(in_channels=dim,
                                              out_channels=dim,
                                              reduction=reduction,
                                              norm_layer=norm_layer)
        self.channel_emb2 = jfe(in_channels=dim,
                                              out_channels=dim,
                                              reduction=reduction,
                                              norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        x1 = self.channel_emb1(x1, H, W)
        x2 = self.channel_emb2(x2, H, W)
        return x1, x2

