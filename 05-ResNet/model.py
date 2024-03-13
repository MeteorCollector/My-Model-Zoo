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

class MyResNet(nn.Module):

    def __init__(self, block, block_num_group, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(MyResNet, self).__init__()
        self.in_channel = 64
        self.block = block

        self.top_layers = nn.Sequential(
            nn.Conv2d(3, self.in_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )

        self.conv2 = self.__make_layer(channel=64,  block_num=block_num_group[0], index=2, stride=1)
        self.conv3 = self.__make_layer(channel=128, block_num=block_num_group[1], index=3, stride=2)
        self.conv4 = self.__make_layer(channel=256, block_num=block_num_group[2], index=4, stride=2)
        self.conv5 = self.__make_layer(channel=512, block_num=block_num_group[3], index=5, stride=2)

        self.outpool = nn.AvgPool2d(7)
        self.fc      = nn.Linear(512 * self.block.expansion, num_classes)

    def __make_layer(self, channel, block_num, index, stride=1):
        
        # block:     block
        # channel:   output channels of this group
        # block_num: number of blocks

        stride_list = [stride] + [1] * (block_num - 1)
        # the first strides is 2, the others are ones.
        layers = nn.Sequential()
        for i in range(len(stride_list)):
            layer_name = f"block_{index}_{i}"
            layers.add_module(layer_name, self.block(self.in_channel, channel, stride_list[i]))
            self.channel = channel * self.block.expansion
        return layers
    
    def forward(self, x):
        
        y = self.top_layers(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.outpool(y)
        y = torch.flatten(y, 0)
        y =self.fc(y)
        y = nn.functional.softmax(y)
        return y
