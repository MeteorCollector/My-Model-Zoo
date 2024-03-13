from torch.nn import Module
from torch import nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = nn.functional.relu(y)
        return y


class BasicBlock(nn.Module):
    # output channel / input channel
    # expands the shortcut
    expansion = 1
    # downsample: a 1x1 convolution
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        #     input (64channel)
        #       +----------+
        # 3x3, 64channel   |
        #       |ReLU      |
        # 3x3, 64channel   |
        #       |          |
        #       +----------+
        #       |ReLU
        #    output

        self.conv1 = ConvBlock(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            # no ReLU here
        )

        self.shortcut = nn.Sequential()

        if stride is not 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        y += self.shortcut(x)
        y = nn.functional.relu(y)
        
        return y

class BottleNeck(nn.Module):
    # output channel / input channel
    # expands the shortcut
    expansion = 4
    # downsample: a 1x1 convolution
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()

        #     input (256channel)
        #       +----------+
        # 1x1, 64channel   |
        #       |ReLU      |
        # 3x3, 64channel   |
        #       |ReLU      |
        # 1x1, 256channel  |
        #       |          |
        #       +----------+
        #       |ReLU
        #    output

        self.conv1 = ConvBlock(in_channel, out_channel, 1, stride=1, padding=1, bias=False)

        self.conv2 = ConvBlock(out_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * self.expansion, 1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            # no ReLU here
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * self.expansion, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
        )

    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y += self.shortcut(x)
        y = nn.functional.relu(y)
        
        return y

class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.in_channel = 64

    def __make_layer(self, block, channel, block_num, stride=1):
        
        downsample = None
        if stride != 1 or self.in_channel != channel + block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, 1, stride=stride)
            )
        
        layers = []
        layers.append(block(self.in_channel, channel, downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        
        return nn.Sequential(*layers)
    
    def forward(x):
        return x