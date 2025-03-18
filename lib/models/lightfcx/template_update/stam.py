import torch.nn as nn

from lib.models.lightfcx.blocks.ffn import integration
from lib.models.lightfcx.blocks.spatial_crossattn import crossattn_spatial

from lib.utils.registry import MODEL_REGISTRY
from lib.utils.token_utils import patch2token, token2patch


@MODEL_REGISTRY.register()
class STAM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.cross = crossattn_spatial(dim, 1)
        self.linear1 = integration(dim)
        self.linear2 = integration(dim)

        self.linear_stg2 = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.ln = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        x1, x2 = patch2token(x1), patch2token(x2)
        x1_res, x2_res = x1, x2
        x1, x2 = self.cross(x1, x2)
        x1, x2 = x1 + x1_res, x2 + x2_res

        x1 = patch2token(self.linear1(token2patch(x1))) + x1_res
        x2 = patch2token(self.linear2(token2patch(x2))) + x2_res

        return token2patch(self.ln(self.linear_stg2(x1 + x2) + x1 + x2))


