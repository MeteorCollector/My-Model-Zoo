from torch import nn
from torchvision import models

def vggs():  
    '''
    由于之前已经搓过vgg了，这里就不重新造轮子了。
    直接调用torchvision.models里面的vgg，
    修改对应的网络层，同样可以得到目标的backbone。
    '''
    vgg16 = models.vgg16()
    vggs = vgg16.features
    vggs[16] = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=True)
    vggs[-1] = nn.MaxPool2d(3, 1, 1, 1, ceil_mode=False)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    '''
    方法一：
    '''
    #vggs= nn.Sequential(feature, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True))

    '''
    方法二：
    '''
    vggs.add_module('31', conv6)
    vggs.add_module('32', nn.ReLU(inplace=True))
    vggs.add_module('33', conv7)
    vggs.add_module('34', nn.ReLU(inplace=True))

    return vggs

    # when called, use this format:
    # vgg = nn.Sequential(*vggs())
    # x = torch.randn(1,3,300,300)
    # print(vgg(x).shape)