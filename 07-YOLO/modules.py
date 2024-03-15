from torch.nn import Module
import torch.nn.functional as F
from torch import nn
import torch

# conv block, as usual.

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, kernel_size, **kwargs)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y

class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBlock(  3,  64, 7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv2 = nn.Sequential(
            ConvBlock( 64, 192, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv3 = nn.Sequential(
            ConvBlock(192, 128, 1, stride=1, padding=0, bias=False),
            ConvBlock(128, 256, 3, stride=1, padding=1, bias=False),
            ConvBlock(256, 256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256, 512, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv4 = nn.Sequential(
            ConvBlock(512,  256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256,  512, 3, stride=1, padding=1, bias=False),
            ConvBlock(512,  256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256,  512, 3, stride=1, padding=1, bias=False),
            ConvBlock(512,  256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256,  512, 3, stride=1, padding=1, bias=False),
            ConvBlock(512,  256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256,  512, 3, stride=1, padding=1, bias=False),
            ConvBlock(512,  256, 1, stride=1, padding=0, bias=False),
            ConvBlock(256,  512, 3, stride=1, padding=1, bias=False),
            ConvBlock(512,  512, 1, stride=1, padding=0, bias=False),
            ConvBlock(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv5 = nn.Sequential(
            ConvBlock(1024,  512, 1, stride=1, padding=0, bias=False),
            ConvBlock( 512, 1024, 3, stride=1, padding=1, bias=False),
            ConvBlock(1024,  512, 1, stride=1, padding=0, bias=False),
            ConvBlock( 512, 1024, 3, stride=1, padding=1, bias=False),
            ConvBlock(1024, 1024, 3, stride=1, padding=1, bias=False),
            ConvBlock(1024, 1024, 3, stride=2, padding=1, bias=False),
        )

        self.conv6 = nn.Sequential(
            ConvBlock(1024, 1024, 3, stride=1, padding=1, bias=False),
            ConvBlock(1024, 1024, 3, stride=1, padding=1, bias=False),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        return y




if __name__ == "__main__":
    # you can test network structure here
    import torch
    x = torch.randn([1, 3, 448, 448])

    net = ConvModule()
    print(net)
    out = net(x)
    print(out.size())