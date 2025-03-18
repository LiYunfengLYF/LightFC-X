import torch.nn as nn
from .centerhead import conv, Center_Head_with_SE
from lib.utils.registry import MODEL_REGISTRY
from .repconv import RepN33
from .se import SE


@MODEL_REGISTRY.register()
class RepN33_SE_Center_Concat(Center_Head_with_SE):
    def __init__(self, inplanes=64, channel=96, feat_sz=20, stride=16, freeze_bn=False):
        super().__init__()

        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        # corner predict
        self.conv1_ctr = RepN33(inplanes, channel)
        self.conv2_ctr = conv(channel, channel // 2, )
        self.conv3_ctr = conv(channel // 2, channel // 4, )
        self.conv4_ctr = conv(channel // 4, channel // 8, )
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.se_ctr = SE(channel, reduction=4)

        # size regress
        self.conv1_offset = RepN33(inplanes, channel, )
        self.conv2_offset = conv(channel, channel // 2, )
        self.conv3_offset = conv(channel // 2, channel // 4, )
        self.conv4_offset = conv(channel // 4, channel // 8, )
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)
        self.se_offset = SE(channel, reduction=4)

        # size regress
        self.conv1_size = RepN33(inplanes, channel, )
        self.conv2_size = conv(channel, channel // 2, )
        self.conv3_size = conv(channel // 2, channel // 4, )
        self.conv4_size = conv(channel // 4, channel // 8, )
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)
        self.se_size = SE(channel, reduction=4)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
