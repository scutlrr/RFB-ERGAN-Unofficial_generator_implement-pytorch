import torch
import torch.nn as nn


# RRDB
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Dense Residual Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, in_planes, gc=32, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_planes, out_planes=gc, non_linearity=True):
            layers = [nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b0 = block(in_planes=in_planes)
        self.b1 = block(in_planes=in_planes + 1 * gc)
        self.b2 = block(in_planes=in_planes + 2 * gc)
        self.b3 = block(in_planes=in_planes + 3 * gc)
        self.b4 = block(in_planes=in_planes + 4 * gc, out_planes=in_planes, non_linearity=False)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(torch.cat((x, x0), 1))
        x2 = self.b2(torch.cat((x, x0, x1), 1))
        x3 = self.b3(torch.cat((x, x0, x1, x2), 1))
        x4 = self.b4(torch.cat((x, x0, x1, x2, x3), 1))
        return x4 * self.res_scale + x


class RRDB(nn.Module):
    def __init__(self, in_planes, res_scale=0.2, num_dense_block=3):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(*[DenseResidualBlock(in_planes) for _ in range(num_dense_block)])

    def forward(self, x):
        return self.dense_blocks(x) * self.res_scale + x


# RRFDB
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0, 0), dilation=1, group=1,
                 non_linearity=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=group, bias=bias)
        self.relu = nn.ReLU(inplace=True) if non_linearity else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, res_scale=0.2, non_linearity=True):
        super(BasicRFB, self).__init__()
        self.out_channels = out_planes
        self.res_scale = res_scale
        self.non_linearity = non_linearity
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, padding=1, non_linearity=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, padding=3, dilation=3, non_linearity=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, padding=3, dilation=3, non_linearity=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, padding=5, dilation=5, non_linearity=False)
        )

        self.ConvLinear = BasicConv(inter_planes * 4, out_planes, kernel_size=1, non_linearity=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, non_linearity=False)
        self.lrelu = nn.LeakyReLU(0.2) if self.non_linearity else None

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        shortcut = self.shortcut(x)
        out = shortcut * self.res_scale + out
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out


class RFDB(nn.Module):
    def __init__(self, in_planes, gc=32, res_scale=0.2):
        super(RFDB, self).__init__()
        self.res_scale = res_scale
        self.rfdb0 = BasicRFB(in_planes, gc)
        self.rfdb1 = BasicRFB(in_planes + gc, gc)
        self.rfdb2 = BasicRFB(in_planes + 2 * gc, gc)
        self.rfdb3 = BasicRFB(in_planes + 3 * gc, gc)
        self.rfdb4 = BasicRFB(in_planes + 4 * gc, in_planes, non_linearity=False)

    def forward(self, x):
        x0 = self.rfdb0(x)
        x1 = self.rfdb1(torch.cat((x, x0), 1))
        x2 = self.rfdb2(torch.cat((x, x0, x1), 1))
        x3 = self.rfdb3(torch.cat((x, x0, x1, x2), 1))
        x4 = self.rfdb4(torch.cat((x, x0, x1, x2, x3), 1))
        return x4 * self.res_scale + x


class RRFDB(nn.Module):
    def __init__(self, in_planes, res_scale=0.2, num_dense_block=3):
        super(RRFDB, self).__init__()
        self.res_scale = res_scale
        self.dense_block = nn.Sequential(*[RFDB(in_planes) for _ in range(num_dense_block)])

    def forward(self, x):
        return self.dense_block(x) * self.res_scale + x


# Generator
class Generator(nn.Module):
    def __init__(self, in_planes, filter, num_res_block_rrdb=16, num_res_block_rrfdb=8, num_upsample=2):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(in_planes, filter, kernel_size=3, stride=1, padding=1)
        # RRDB blocks
        self.trunk_a = nn.Sequential(*[RRDB(filter) for _ in range(num_res_block_rrdb)])
        # RRFDB blocks
        self.trunk_rfb = nn.Sequential(*[RRFDB(filter) for _ in range(num_res_block_rrfdb)])
        # RFB layer post RRFDB blocks
        self.rfb = BasicRFB(filter, filter, non_linearity=False)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                BasicRFB(filter, filter),
                nn.Conv2d(filter, filter * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
                BasicRFB(filter, filter)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filter, in_planes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.trunk_a(out1)
        out2 = self.trunk_rfb(out2)
        out = torch.add(out1, out2)
        out = self.rfb(out)
        out = self.upsampling(out)
        out = self.conv2(out)
        return out


if __name__ == '__main__':
    input=torch.randn(1,3,128,128)
    generator=Generator(3,64,num_res_block_rrdb=2,num_res_block_rrfdb=2,num_upsample=1)
    # print(generator)
    output=generator(input)
    print(output.size())