import torch.nn as nn

class jfe(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv11_up = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True)
        self.conv33 = nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1,
                                padding=1, bias=True, groups=out_channels // reduction)
        self.conv55 = nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=5, stride=1,
                                padding=2, bias=True, groups=out_channels // reduction)
        self.conv77 = nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=7, stride=1,
                                padding=3, bias=True, groups=out_channels // reduction)
        self.act = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True)
        self.norm1 = norm_layer(out_channels)

        self.norm2 = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.conv11_up(x)
        x = self.conv33(x) + self.conv55(x) + self.conv77(x)
        x = self.act(x)
        x = self.conv11(x)
        x = self.norm1(x)
        out = self.norm2(residual + x)
        return out
