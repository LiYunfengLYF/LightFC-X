import torch.nn as nn



class dwconv33(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                                            groups=dim),
                                  nn.BatchNorm2d(dim))

    def forward(self, x):
        return self.conv(x)


class dwconv55(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                            groups=dim),
                                  nn.BatchNorm2d(dim))

    def forward(self, x):
        return self.conv(x)


class dwconv77(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3,
                                            groups=dim),
                                  nn.BatchNorm2d(dim))

    def forward(self, x):
        return self.conv(x)


class convlinear(nn.Module):
    def __init__(self, dim, factor=2):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * factor, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(in_channels=dim * factor, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.channel(x)


class convlinear_rmconv(nn.Module):
    def __init__(self, dim, factor=2):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU()
        )

    def forward(self, x):
        return self.channel(x)


class repconvlinear(nn.Module):
    def __init__(self, dim, factor=2):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * factor, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(in_channels=dim * factor, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.channel(x)


class integration(nn.Module):
    def __init__(self, dim, factor=2):
        super().__init__()
        self.conv = dwconv33(dim)
        self.linear = convlinear(dim, factor)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.linear(x) + x
        return x
