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


class InceptionBlock(Module):
    def __init__(self, in_channels, out_1=24, chn_3=16, out_3=24, chn_5=16, out_5=24, out_pool=24):
        super(InceptionBlock, self).__init__()

        #              output
        #
        #                         3x3conv
        #                           (24)
        #
        # 1x1conv         5x5conv 3x3conv
        #  (24)             (24)    (24)
        #
        # Average 1x1conv 1x1conv 1x1conv
        # Pooling  (16)     (16)    (16)
        #
        #              input

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

        self.branch_1 = nn.Sequential(
            ConvBlock(in_channels, out_1, kernel_size=1),
        )

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, chn_5, kernel_size=1),
            ConvBlock(chn_5, out_5, kernel_size=5, padding=2),
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, chn_3, kernel_size=1),
            ConvBlock(chn_3, out_3, kernel_size=3, padding=1),
            ConvBlock(out_3, out_3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):

        out_pool = self.branch_pool(x)
        out_1    = self.branch_1(x)
        out_2    = self.branch_2(x)
        out_3    = self.branch_3(x)

        # print(out_pool.size())
        # print(out_1.size())
        # print(out_2.size())
        # print(out_3.size())


        y = [out_1, out_2, out_3, out_pool]

        return torch.cat(y, dim=1)


class MyInceptionNet(Module):
    
    def __init__(self, num_classes):
        super(MyInceptionNet, self).__init__()
        
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        # using id here to match original paper
        self.inception_3a = InceptionBlock(192,  64,  96, 128,  16,  32,  32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192,  32,  96,  64)
        self.inception_4a = InceptionBlock(480, 192,  96, 208,  16,  48,  64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224,  24,  64,  64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256,  24,  64,  64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288,  32,  64,  64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320,  32, 128, 128)
        self.inception_5a = InceptionBlock(832, 256, 160, 320,  32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384,  48, 128, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fullcnc = nn.Linear(1024, num_classes)


    def forward(self, x):
        
        y = self.conv1(x)
        y = self.maxpool(y)
        y = self.conv2(y)
        y = self.maxpool(y)

        y = self.inception_3a(y)
        y = self.inception_3b(y)
        y = self.maxpool(y)
        y = self.inception_4a(y)

        # original net has aux0 here
        # but I don't want to add this

        y = self.inception_4b(y)
        y = self.inception_4c(y)
        y = self.inception_4d(y)

        # original net has aux1 here
        # but I don't want to add this

        y = self.inception_4e(y)
        y = self.maxpool(y)
        y = self.inception_5a(y)
        y = self.inception_5b(y) 
        y = self.avgpool(y)

        y = torch.flatten(y, 1)
        y = self.dropout(y)
        y = self.fullcnc(y)

        # original net has aux2 here
        # but I don't want to add this
        
        return y
