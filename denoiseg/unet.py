import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        depth: int = 3,
        start_filters: int = 16,
        up_mode: str = "transposed",
        dropout=0.2,
    ):
        super().__init__()

        self.inc = ConvLayer(in_channels, start_filters, dropout=dropout)
        self.depth = depth
        # Contracting path
        self.down = nn.ModuleList(
            [
                DownSamplingLayer(
                    start_filters * 2**i, start_filters * 2 ** (i + 1), dropout=dropout
                )
                for i in range(depth)
            ]
        )

        self.drop = nn.Dropout(dropout)
        # Expansive path
        self.up = nn.ModuleList(
            [
                UpSamplingLayer(
                    start_filters * 2 ** (i + 1),
                    start_filters * 2**i,
                    up_mode,
                    dropout=dropout,
                )
                for i in range(depth - 1, -1, -1)
            ]
        )

        self.outc = nn.Sequential(
            nn.Conv2d(start_filters, out_channels, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.inc(x)

        outputs = []

        for module in self.down:
            outputs.append(x)
            x = module(x)
        x = self.drop(x)

        for module, output in zip(self.up, outputs[::-1]):
            x = module(x, output)

        return self.outc(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 2, dropout=0.2):
        super().__init__()

        layers = []
        for i in range(n):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.Dropout(p=dropout))

        layers.pop()  # remove last dropout

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout=0.2):
        super().__init__()

        self.layer = nn.Sequential(
            nn.MaxPool2d(2), ConvLayer(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, mode: str = "transposed", dropout=0.2
    ):
        """
        :param mode: 'transposed' for transposed convolution, or 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        """
        super().__init__()

        if mode == "transposed":
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
                nn.Conv2d(
                    in_channels, in_channels // 2, kernel_size=1, dropout=dropout
                ),
            )

        self.conv = ConvLayer(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
