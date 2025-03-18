import torch.nn as nn
from lib.utils.registry import MODEL_REGISTRY

from lib.models.lightfcx.template_update.crosattn import crosattn_spatial_q2k1v2
from lib.utils.token_utils import token2patch


@MODEL_REGISTRY.register()
class directupdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return (x1 + x2) / 2


@MODEL_REGISTRY.register()
class directupdate_crosattn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.crossattn = crosattn_spatial_q2k1v2(dim)

    def forward(self, x1, x2):
        x21 = self.crossattn(x1, x2)
        return (x1 + token2patch(x21)) / 2
