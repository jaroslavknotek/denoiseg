

import torch
from torch import nn


class UNetAttention(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        depth: int = 3,
        start_filters: int = 16,
        up_mode: str = "transposed",
        dropout=0.2,
        use_attention = False
    ):
        super().__init__()

        self.inc = ConvLayer(in_channels, start_filters, dropout=dropout)

        # Contracting path
        self.down = nn.ModuleList(
            [
                DownSamplingLayer(
                    start_filters * 2**i, 
                    start_filters * 2 ** (i + 1), 
                    dropout=dropout
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
                    use_attention = use_attention
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
            nn.MaxPool2d(2), 
            ConvLayer(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.layer(x)


class UpSamplingLayer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mode: str = "transposed", 
        dropout=0.2,
        use_attention = False,
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
                    in_channels, 
                    in_channels // 2, 
                    kernel_size=1, 
                    dropout=dropout
                ),
            )
        
        self.attention = None
        if use_attention:
            self.attention = Attn(in_channels//2)

        self.conv = ConvLayer(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.attention is not None:
            x2 = self.attention(x2, x1)
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#     ###

# class Conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#     def forward(self, x):
#             return self.conv(x)


# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, is_res=True):
#         super(ConvBlock, self).__init__()
#         self.conv1 = Conv(in_ch, out_ch)
#         self.conv2 = Conv(out_ch, out_ch)
#         self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.activation = nn.ReLU()
#         self.is_res = is_res
#     def forward(self, x):
#         x = self.conv1(x)
#         y = self.conv2(x)
#         y = self.conv3(y)
#         y = self.bn(y)
#         if self.is_res:
#             y += x
#         return self.activation(y)


# class DeconvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, attn=None):
#         super(DeconvBlock, self).__init__()
#         self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
#         self.conv = ConvBlock(in_ch, out_ch)
#         self.attn = attn
        
#     def forward(self, x, bridge):
#         x = self.deconv(x)
#         if self.attn:
#             bridge = self.attn(bridge, x)
#         x = torch.cat([x, bridge], dim=1)
#         return self.conv(x)


# class Encoder(nn.Module):
#     def __init__(self, in_ch=3, out_ch=64, depth=5):
#         super(Encoder, self).__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.convs = nn.ModuleList()
#         for _ in range(depth):
#             self.convs.append(ConvBlock(in_ch, out_ch))
#             in_ch = out_ch
#             out_ch *= 2
#     def forward(self, x):
#         res = []
#         for i, m in enumerate(self.convs):
#             if i > 0:
#                 x = self.pool(x)
#             x = m(x)
#             res.append(x)
#         return res


class Attn(nn.Module):
        '''
        Attention U-Net: Learning Where to Look for the Pancreas
        https://arxiv.org/pdf/1804.03999.pdf
        '''
        def __init__(self, ch):
            super(Attn, self).__init__()
            self.wx = nn.Conv2d(ch, ch, 1)
            self.wg = nn.Conv2d(ch, ch, 1)
            self.psi = nn.Conv2d(ch, ch, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.ch = ch
            
        def forward(self, x, g):
            identity = x
            x = self.wx(x)
            g = self.wg(g)
            x = self.relu(x + g)
            x = self.psi(x)
            x = self.sigmoid(x)
            return identity * (x + 1)


# class Decoder(nn.Module):
#     def __init__(self, in_ch=1024, depth=4, attn=True):
#         super(Decoder, self).__init__()
#         self.depth = depth
#         self.deconvs = nn.ModuleList()
#         for _ in range(depth):
#             self.deconvs.append(
#                 DeconvBlock(
#                     in_ch, 
#                     in_ch // 2, 
#                     Attn(in_ch // 2) if attn else None)
#             )
#             in_ch //= 2

#     def forward(self, x_list):
#         for i in range(self.depth):
#             if i == 0:
#                 x = x_list.pop()
#             bridge = x_list.pop()
#             x = self.deconvs[i](x, bridge)
#         return x

# class UNet(nn.Module):
#     '''
#     U-Net: Convolutional Networks for Biomedical Image Segmentation
#     https://arxiv.org/pdf/1505.04597.pdf
#     '''
#     def __init__(
#         self, 
#         in_ch=3, 
#         out_ch=1, 
#         encoder_depth=5, 
#         regressive=False, 
#         attn=True
#     ):
#         super(UNet, self).__init__()
#         self.encoder = Encoder(in_ch, 64, encoder_depth)
#         self.decoder = Decoder(1024, encoder_depth-1, attn)
#         self.conv = nn.Conv2d(64, out_ch, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.regressive = regressive
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = self.conv(x)
#         if self.regressive:
#             return x
#         else:
#             return self.sigmoid(x).clamp(1e-4, 1 - 1e-4)
