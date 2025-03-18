import torch
import torch.nn as nn

from lib.utils.token_utils import patch2token, token2patch
from .ecm import pixel_wise_corr, SE
from lib.utils.registry import MODEL_REGISTRY
from ..blocks.ffm import ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357, \
    ffm_crospath_cat2add_linear2conv1_croattn_spatial_relu_ffn_357_stage2_unshare


@MODEL_REGISTRY.register()
class ECM_ECAM(nn.Module):

    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()

        # pw-corr
        self.pw_corr = pixel_wise_corr
        self.ca = SE(reduction=1, channels=num_kernel)

        # SCF
        self.conv33 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=3, stride=1, padding=1,
                                groups=num_kernel)
        self.bn33 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=1, stride=1, padding=0,
                                groups=num_kernel)
        self.bn11 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

        # RGB-T fusion
        self.rgbt_ffm = ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357(dim=96, num_heads=1)

    def forward(self, z, x):
        _, C, _, _ = z.shape
        z_v, z_i = torch.split(z, (int(C / 2), int(C / 2),), dim=1)
        x_v, x_i = torch.split(x, (int(C / 2), int(C / 2),), dim=1)

        # v2v
        corr_v2v = self.ca(self.pw_corr(z_v, x_v))
        corr_v2v = corr_v2v + self.bn11(self.conv11(corr_v2v)) + self.bn33(self.conv33(corr_v2v))
        corr_v2v = corr_v2v + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_v2v)))))
        corr_v2v = self.adjust(corr_v2v)

        # v2i
        corr_i2i = self.ca(self.pw_corr(z_i, x_i))
        corr_i2i = corr_i2i + self.bn11(self.conv11(corr_i2i)) + self.bn33(self.conv33(corr_i2i))
        corr_i2i = corr_i2i + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_i2i)))))
        corr_i2i = self.adjust(corr_i2i)

        # cmx ffm
        corr = self.rgbt_ffm(corr_v2v, corr_i2i)

        corr = torch.cat((corr, x), dim=1)

        return corr


@MODEL_REGISTRY.register()
class ECM_3xECAM(nn.Module):
    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()

        # pw-corr
        self.pw_corr = pixel_wise_corr
        self.ca = SE(reduction=1, channels=num_kernel)

        # SCF
        self.conv33 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=3, stride=1, padding=1,
                                groups=num_kernel)
        self.bn33 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=1, stride=1, padding=0,
                                groups=num_kernel)
        self.bn11 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

        # RGB-T fusion
        self.rgbt_ffm = ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357(dim=96, num_heads=1)
        self.rgbt_ffm2 = ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357(dim=96, num_heads=1)
        self.rgbt_ffm3 = ffm_crospath_cat2add_linear2conv1_croattn_spatial_ffn_357(dim=96, num_heads=1)

    def forward(self, z, x):
        _, C, _, _ = z.shape
        z_v, z_i = torch.split(z, (int(C / 2), int(C / 2),), dim=1)
        x_v, x_i = torch.split(x, (int(C / 2), int(C / 2),), dim=1)

        # v2v
        corr_v2v = self.ca(self.pw_corr(z_v, x_v))
        corr_v2v = corr_v2v + self.bn11(self.conv11(corr_v2v)) + self.bn33(self.conv33(corr_v2v))
        corr_v2v = corr_v2v + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_v2v)))))
        corr_v2v = self.adjust(corr_v2v)

        # v2i
        corr_i2i = self.ca(self.pw_corr(z_i, x_i))
        corr_i2i = corr_i2i + self.bn11(self.conv11(corr_i2i)) + self.bn33(self.conv33(corr_i2i))
        corr_i2i = corr_i2i + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_i2i)))))
        corr_i2i = self.adjust(corr_i2i)

        # cmx ffm
        corr = self.rgbt_ffm(corr_v2v, corr_i2i)
        corr_v2v = corr + corr_v2v
        corr_i2i = corr_i2i + corr_i2i
        corr = self.rgbt_ffm2(corr_v2v, corr_i2i)

        corr_v2v = corr + corr_v2v
        corr_i2i = corr_i2i + corr_i2i
        corr = self.rgbt_ffm3(corr_v2v, corr_i2i)

        corr = torch.cat((corr, x), dim=1)

        return corr


@MODEL_REGISTRY.register()
class ECM_ECAM_for_RGBS(nn.Module):

    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()

        # pw-corr
        self.pw_corr = pixel_wise_corr
        self.ca = SE(reduction=1, channels=num_kernel)

        # SCF
        self.conv33 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=3, stride=1, padding=1,
                                groups=num_kernel)
        self.bn33 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=1, stride=1, padding=0,
                                groups=num_kernel)
        self.bn11 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

        # RGB-S fusion
        self.rgbs_ffm = ffm_crospath_cat2add_linear2conv1_croattn_spatial_relu_ffn_357_stage2_unshare(dim=96,
                                                                                                      num_heads=1)

    def forward(self, z, x):
        # print(z[0].shape)
        _, C, _, _ = z.shape
        z_v, z_s = torch.split(z, (int(C / 2), int(C / 2),), dim=1)
        x_v, x_s = torch.split(x, (int(C / 2), int(C / 2),), dim=1)
        #
        # z_v, z_s = z[0], z[1]
        # x_v, x_s = x[0], x[1]

        # v2v
        corr_v2v = self.ca(self.pw_corr(z_v, x_v))
        corr_v2v = corr_v2v + self.bn11(self.conv11(corr_v2v)) + self.bn33(self.conv33(corr_v2v))
        corr_v2v = corr_v2v + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_v2v)))))
        corr_v2v = self.adjust(corr_v2v)

        # s2s
        corr_s2s = self.ca(self.pw_corr(z_s, x_s))
        corr_s2s = corr_s2s + self.bn11(self.conv11(corr_s2s)) + self.bn33(self.conv33(corr_s2s))
        corr_s2s = corr_s2s + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr_s2s)))))
        corr_s2s = self.adjust(corr_s2s)

        # cmx ffm
        corr1, corr2 = self.rgbs_ffm(corr_v2v, corr_s2s)

        corr_v = torch.cat((corr1, x_v), dim=1)
        corr_s = torch.cat((corr2, x_s), dim=1)
        return corr_v, corr_s
