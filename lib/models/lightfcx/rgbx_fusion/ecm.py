import torch
import torch.nn as nn

from lib.utils.registry import MODEL_REGISTRY


def pixel_wise_corr(z, x):
    """
    z is kernel ([B, C, 8, 8])
    x is search ([B, C, 16, 16])

    z -> (B, 64, C)
    x -> (B, C, 256)
    """
    b, c, h, w = x.size()
    z_mat = z.contiguous().view((b, c, -1)).transpose(1, 2)  # (b,64,c)
    x_mat = x.contiguous().view((b, c, -1))  # (b,c,256)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))


class SE(nn.Module):

    def __init__(self, channels=64, reduction=1):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


@MODEL_REGISTRY.register()
class PWCorr_SE_SCF31_IAB11_Concat_Release(nn.Module):
    """
    Now it is used and released
    """

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

    def forward(self, z, x):
        corr = self.ca(self.pw_corr(z, x))

        # scf + skip-connection
        corr = corr + self.bn11(self.conv11(corr)) + self.bn33(self.conv33(corr))

        # iab + skip-connection
        corr = corr + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr)))))

        corr = self.adjust(corr)

        corr = torch.cat((corr, x), dim=1)

        return corr
