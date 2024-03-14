from torch.nn import Module
from torch import nn
import torch
from math import sqrt
from vgg import vgg
from layers import PriorBox
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    # L2 Normalize
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out



def ExtraFeature():
    # combine features of different scales
    conv8_1  = nn.Conv2d(1024, 256, 1, stride=1)
    conv8_2  = nn.Conv2d( 256, 512, 3, stride=2, padding=1)
    conv9_1  = nn.Conv2d( 512, 128, 1, stride=1)
    conv9_2  = nn.Conv2d( 128, 256, 3, stride=2, padding=1)
    conv10_1 = nn.Conv2d( 256, 128, 1, stride=1)
    conv10_2 = nn.Conv2d( 128, 256, 3, stride=1)
    conv11_1 = nn.Conv2d( 256, 128, 1, stride=1)
    conv11_2 = nn.Conv2d( 128, 256, 3, stride=1)
    
    return [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]

def MultiBox(vgg, extras, num_classes):

    # reference in paper:
    #
    # We initialize the parameters
    # for all the newly added convolutional layers 
    # with the ”xavier” method [20]. 
    # For conv4_3, conv10_2 and conv11_2,
    # we only associate 4 default boxes at each feature map location
    # – omitting aspect ratios of 1/3 and 3.
    # For all other layers, we put 6 default boxes as
    # described in Sec. 2.2. 

    loc_layers  = []
    conf_layers = []

    # location prediction (bounding-box regression) layer
    loc1 = nn.Conv2d(  vgg[21].out_channels, 4 * 4, 3, stride=1) # conv4_3 
    loc2 = nn.Conv2d(  vgg[-2].out_channels, 6 * 4, 3, stride=1) # conv7
    loc3 = nn.Conv2d(extras[1].out_channels, 6 * 4, 3, stride=1) # exts1_2
    loc4 = nn.Conv2d(extras[3].out_channels, 6 * 4, 3, stride=1) # exts2_2
    loc5 = nn.Conv2d(extras[5].out_channels, 4 * 4, 3, stride=1) # exts3_2
    loc6 = nn.Conv2d(extras[7].out_channels, 4 * 4, 3, stride=1) # exts4_2
    loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]

    # classifier layer
    conf1 = nn.Conv2d(  vgg[21].out_channels, 4 * num_classes, 3, stride=1) # conv4_3 
    conf2 = nn.Conv2d(  vgg[-2].out_channels, 6 * num_classes, 3, stride=1) # conv7
    conf3 = nn.Conv2d(extras[1].out_channels, 6 * num_classes, 3, stride=1) # exts1_2
    conf4 = nn.Conv2d(extras[3].out_channels, 6 * num_classes, 3, stride=1) # exts2_2
    conf5 = nn.Conv2d(extras[5].out_channels, 4 * num_classes, 3, stride=1) # exts3_2
    conf6 = nn.Conv2d(extras[7].out_channels, 4 * num_classes, 3, stride=1) # exts4_2
    conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]

    return loc_layers, conf_layers



class MySSD(nn.Module):

    # reference: https://hellozhaozheng.github.io/z_post/PyTorch-SSD/#build_ssd

    # SSD网络是由 VGG 网络后接 multibox 卷积层 组成的, 每一个 multibox 层会有如下分支:
    # - 用于class conf scores的卷积层
    # - 用于localization predictions的卷积层
    # - 与priorbox layer相关联, 产生默认的bounding box

    # 参数:
    # phase: test/train
    # size: 输入图片的尺寸
    # base: VGG16的层
    # extras: 将输出结果送到multibox loc和conf layers的额外的层
    # head: "multibox head", 包含一系列的loc和conf卷积层.

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(MySSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # 如果num_classes=21，则选择coco，否则选择voc
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        # from torch.autograd import Variable
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        # head = (loc_layers, conf_layers)
        self.loc = nn.ModuleList(head[0]) 
        self.conf = nn.ModuleList(head[1])


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

