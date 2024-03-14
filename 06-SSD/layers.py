import torch
from torch.nn import Module
from torch import nn
from math import sqrt
from itertools import product
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
from boxutils import decode, nms, match, log_sum_exp
from data import voc as cfg

class PriorBox(object):
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=21]
        # 如果num_classes=21，则选择coco，否则选择voc
        # coco, voc的相关配置都来自于data/cfg.py 文件
        super(PriorBox, self).__init__()
        
        # 读取config文件
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")
    
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): 
            # 存放的是feature map的尺寸:38,19,10,5,3,1
            # from itertools import product as product
            for i, j in product(range(f), repeat=2):
                # for all pairs if (i, j) in range(f)
                # 产生anchor的坐标(i,j)

                f_k = self.image_size / self.steps[k]
                # steps=[8,16,32,64,100,300]. f_k为feature map的尺寸(相对)
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该cy与i对应.

                # aspect_ratios 为1时对应的box
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改.
        output = torch.Tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1, min=0) # clamp_ 是clamp的原地执行版本
        return output # 输出default box坐标(可以理解为anchor box)



class Detect(Function):
    # 测试阶段的最后一层, 负责解码预测结果, 应用nms选出合适的框和对应类别的置信度.
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = cfg["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        # loc_data: [batch, num_priors, 4], [batch, 8732, 4]
        # conf_data: [batch, num_priors, 21], [batch, 8732, 21]
        # prior_data: [num_priors, 4], [8732, 4]

        num = loc_data.size(0) # batch_size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5) # output:[b, 21, k, 5]
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2,1) # 维度调换

        # 将预测结果解码
        for i in range(num): # 对每一个image进行解码
            decoded_boxes = decode(loc_data[i], prior_data, self.variance) #获取第i个图片的box坐标
            conf_scores = conf_preds[i].clone() # 复制第i个image置信度预测结果

            for cl in range(1, self.num_classes): # num_classes=21, 所以 cl 的值为 1~20
                c_mask = conf_scores[cl].gt(self.conf_thresh) # 返回由0,1组成的数组, 0代表小于thresh, 1代表大于thresh
                scores = conf_scores[cl][c_mask] # 返回值为1的对应下标的元素值(即返回conf_scores中大于thresh的元素集合)

                if scores.size(0) == 0:
                    continue # 没有置信度, 说明没有框
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) # 获取对应box的二值矩阵
                boxes = decoded_boxes[l_mask].view(-1,4) # 获取置信度大于thresh的box的左上角和右下角坐标

                # 返回每个类别的最高的score 的下标, 并且除去那些与该box有较大交并比的box
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k) # 从这些box里面选出top_k个, count<=top_k
                # count<=top_k
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[:count]), 1)
        flt = output.contiguous().view(num,-1,5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # 注意, view共享tensor, 因此, 对flt的修改也会反应到output上面
        return output

class MultiBoxLoss(nn.Module):
    # 计算目标:
    # 输出那些与真实框的iou大于一定阈值的框的下标.
    # 根据与真实框的偏移量输出localization目标
    # 用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3)
    # 目标损失:
    # L(x,c,l,g) = (Lconf(x,c) + αLloc(x,l,g)) / N
    # 参数:
    # c: 类别置信度(class confidences)
    # l: 预测的框(predicted boxes)
    # g: 真实框(ground truth boxes)
    # N: 匹配到的框的数量(number of matched default boxes)

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes= num_classes # 列表数
        self.threshold = overlap_thresh # 交并比阈值, 0.5
        self.background_label = bkg_label # 背景标签, 0
        self.use_prior_for_matching = prior_for_matching # True 没卵用
        self.do_neg_mining = neg_mining # True, 没卵用
        self.negpos_ratio = neg_pos # 负样本和正样本的比例, 3:1
        self.neg_overlap = neg_overlap # 0.5 判定负样本的阈值.
        self.encode_target = encode_target # False 没卵用
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, 21]
        # priors: [8732, 4]  default box 对于任意的图片, 都是相同的, 因此无需带有 batch 维度
        num = loc_data.size(0) # num = batch_size
        priors = priors[:loc_data.size(1), :] # loc_data.size(1) = 8732, 因此 priors 维持不变
        num_priors = (priors.size(0)) # num_priors = 8732
        num_classes = self.num_classes # num_classes = 21 (默认为voc数据集)

        # 将priors(default boxes)和ground truth boxes匹配
        loc_t = torch.Tensor(num, num_priors, 4) # shape:[batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors) # shape:[batch_size, 8732]
        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量, 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data # [num_objs, 4]
            labels = targets[idx][:, -1].data # [num_objs] 使用的是 -1, 而不是 -1:, 因此, 返回的维度变少了
            defaults = priors.data # [8732, 4]
            # from ..box_utils import match
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx) # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值, 关于 match 的详细讲解可以看后面的代码解析
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 用Variable封装loc_t, 新版本的 PyTorch 无需这么做, 只需要将 requires_grad 属性设置为 True 就行了
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 # 筛选出 >0 的box下标(大部分都是=0的)
        num_pos = pos.sum(dim=1, keepdim=True) # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data:[batch, num_priors, 4]
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)# 获取预测结果值
        loc_t = loc_t[pos_idx].view(-1, 4) # 获取gt值
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # 计算损失

        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_data: [batch, num_priors, num_classes]
        # batch_conf: [batch, num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes) # reshape

        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        loss_c[pos.view(-1, 1)] = 0 # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1) # reshape
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算包括正样本和负样本的置信度损失
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx筛选目标数据
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 将损失函数归一化后返回
        N = num_pos.data.sum()
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c



class MultiBoxLoss(nn.Module):
    # 计算目标:
    # 输出那些与真实框的IoU大于一定阈值的框的下标.
    # 根据与真实框的偏移量输出localization目标
    # 用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3)
    # 目标损失:
    # L(x,c,l,g) = (Lconf(x,c) + αLloc(x,l,g)) / N
    # 参数:
    # c: 类别置信度(class confidences)
    # l: 预测的框(predicted boxes)
    # g: 真实框(ground truth boxes)
    # N: 匹配到的框的数量(number of matched default boxes)

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes= num_classes # 列表数
        self.threshold = overlap_thresh # 交并比阈值, 0.5
        self.background_label = bkg_label # 背景标签, 0
        self.use_prior_for_matching = prior_for_matching # True 没卵用
        self.do_neg_mining = neg_mining # True, 没卵用
        self.negpos_ratio = neg_pos # 负样本和正样本的比例, 3:1
        self.neg_overlap = neg_overlap # 0.5 判定负样本的阈值.
        self.encode_target = encode_target # False 没卵用
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, 21]
        # priors: [8732, 4]  default box 对于任意的图片, 都是相同的, 因此无需带有 batch 维度
        num = loc_data.size(0) # num = batch_size
        priors = priors[:loc_data.size(1), :] # loc_data.size(1) = 8732, 因此 priors 维持不变
        num_priors = (priors.size(0)) # num_priors = 8732
        num_classes = self.num_classes # num_classes = 21 (默认为voc数据集)

        # 将priors(default boxes)和ground truth boxes匹配
        loc_t = torch.Tensor(num, num_priors, 4) # shape:[batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors) # shape:[batch_size, 8732]
        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量, 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data # [num_objs, 4]
            labels = targets[idx][:, -1].data # [num_objs] 使用的是 -1, 而不是 -1:, 因此, 返回的维度变少了
            defaults = priors.data # [8732, 4]
            # from ..box_utils import match
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx) # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值, 关于 match 的详细讲解可以看后面的代码解析
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 用Variable封装loc_t, 新版本的 PyTorch 无需这么做, 只需要将 requires_grad 属性设置为 True 就行了
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 # 筛选出 > 0 的box下标(大部分都是=0的)
        num_pos = pos.sum(dim=1, keepdim=True) # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data:[batch, num_priors, 4]
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)# 获取预测结果值
        loc_t = loc_t[pos_idx].view(-1, 4) # 获取gt值
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # 计算损失

        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_data: [batch, num_priors, num_classes]
        # batch_conf: [batch, num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes) # reshape

        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        loss_c[pos.view(-1, 1)] = 0 # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1) # reshape
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算包括正样本和负样本的置信度损失
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx筛选目标数据
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 将损失函数归一化后返回
        N = num_pos.data.sum()
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c