from torch.nn import Module
from torch import nn
import torch
from math import sqrt
from vgg import vgg
from data import voc, coco
from layers import PriorBox, Detect
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
import os

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

    return vgg, extras, (loc_layers, conf_layers)



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

        if phase == "test":
            self.softmax = nn.Softmax(dim = -1)
            self.detect = Detect(num_classes, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, x):
       
        # 参数: x, 输入的batch 图片, Shape: [batch, 3, 300, 300]
        # 返回值: 取决于不同阶段
        # test: 预测的类别标签, confidence score, 以及相关的location.
        #       Shape: [batch, topk, 7]
        # train: 关于以下输出的元素组成的列表
        #       1: confidence layers, Shape: [batch*num_priors, num_classes]
        #       2: localization layers, Shape: [batch, num_priors*4]
        #       3: priorbox layers, Shape: [2, num_priors*4]

        sources = list() # 用于存储参与预测的卷积层的输出
        # 共有6个，见原论文
        loc  = list()    # 用于存储预测的边框信息
        conf = list()    # 用于存储预测的类别信息

        # 计算vgg直到conv4_3的relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s) # 将 conv4_3 的特征层输出添加到 sources 中, 后面会根据 sources 中的元素进行预测

        # 将vgg应用到fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x) # 同理, 添加到 sources 列表中

        # 计算extra layers, 并且将结果存储到sources列表中
        for k, v in enumerate(self.extras):
            x = nn.functional.relu(v(x), inplace=True) # import torch.nn.functional as F
            if k % 2 == 1: # 在extras_layers中, 第1,3,5,7,9(从第0开始)的卷积层的输出会用于预测box位置和类别, 因此, 将其添加到 sources列表中
                sources.append(x)

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # apply multibox to source layers

        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # 用zip将source, loc, conf对齐分组，进行枚举
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
            # loc:  [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
        
        # 合并这六个尺度的特征图信息
        # 在调用view之前, 需要先调用contiguous，原因在上文已经说明
        # 这里或者用 o.flatten(o, 1) 也没有问题
        loc  = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]

        if self.phase == "test":
            # 这里用到了 detect 对象, 该对象主要由于接预测出来的结果进行解析, 以获得方便可视化的边框坐标和类别编号
            output = self.detect(
                loc.view(loc.size(0), -1, 4), #  又将shape转换成: [batch, num_boxes, 4], 即[1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)), # 同理,  shape 为[batch, num_boxes, num_classes], 即 [1, 8732, 21]
                self.priors(x.data)
                # 利用 PriorBox 对象获取 feature map 上的 default box, 该参数的shape为: [8732,4]. 关于生成 default box 的方法实际上很简单, 类似于 anchor box, 详细的代码实现会在后文解析.
                # 这里原来的的 self.priors.type(type(x.data)) 与 self.priors 就结果而言完全等价(自己试验过了), 但是为什么?
                # 这里源码的可读性差得令人发指，我真的快崩溃了
            )
        if self.phase == "train": # 如果是训练阶段, 则无需解析预测结果, 直接返回然后求损失.
            output = (
                loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
            )
        return output
    
    def load_weights(self, base_file): # 加载权重文件
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights...")
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print("Finished!")
        else:
            print("Error: only .pth and .pkl files supported")



base = { # vgg 网络结构参数
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '500': []
}
extras = { # extras 层参数
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '500': []
}
mbox = { # multibox 相关参数
    '300': [4, 6, 6, 6, 4, 4],
    '500': []
}
def build_ssd(phase, size=300, num_classes=21):
    # 构建模型函数, 调用上面的函数进行构建
    if phase != "test" and phase != "train": # 只能是训练或者预测阶段
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, "+
                "currently only SSD300 is supported!") # 仅仅支持300size的SSD
        return
    base_, extras_, head_ = MultiBox(vgg(base[str(size)], 3),
                                    ExtraFeature,
                                    num_classes)
    
    return MySSD(phase, size, base_, extras_, head_, num_classes)