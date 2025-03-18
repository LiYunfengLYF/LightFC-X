import torch.nn as nn

from lib.utils.token_utils import patch2token, token2patch


class crosattn_spatial_q2k1v2(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x1, x2):
        x1 = patch2token(x1)
        x2 = patch2token(x2)


        B, N, C = x1.shape

        q2 = v2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        ctx2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x2 = (ctx2 @ v2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x2
