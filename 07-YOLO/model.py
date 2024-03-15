from modules import ConvModule
import numpy as np
import torch
import torch.nn as nn
import torchvision
import boxutils

class MyYOLO(nn.Module):
    def __init__(self, S, out_channel):
        """
        grid is the amount of YOLO grids per row/column.
        """

        super(MyYOLO, self).__init__()
        self.backbone = ConvModule()
        # in referenced repo,
        # author used a pretrained res50 as backbone
        # we just observe the original configuration.
        self.grid = S

        self.fc = nn.Sequential(
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, out_channel * S * S)
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = self.backbone(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        y = y.view(batch_size, self.grid ** 2, -1)
        return y

# S denotes the amount of grid per row/column
# B denotes the amount of bounding box detectors
# the equation can be found in original paper
    
class yolo_loss:

    def __init__(self, device, S, B, image_size, num_class):
        self.device = device
        self.S      = S
        self.B      = B
        self.image_size = image_size
        self.num_class  = num_class
        self.batch_size = 0
    
    def __call__(self, input, target):
        """
        :param input: (yolo net output)
                      tensor[s, s, b*5 + n_class] bbox: b * (c_x, c_y, w, h, obj_conf), class1_p, class2_p.. %
        :param target: (dataset) tensor[n_bbox] bbox: x_min, ymin, xmax, ymax, class
        :return: loss tensor

        grid type: [[bbox, ..], [], ..] -> bbox_in_grid: c_x(%), c_y(%), w(%), h(%), class(int)

        target to grid type
        if s = 7 -> grid idx: 1 -> 49
        由于没有使用PyTorch的损失函数，所以需要先分离不同的batch分别计算损失
        """
        self.batch_size = input.size(0)

        # 预处理label，得到的是每张图片的target列表
        target = [self.label_direct2grid(target[i]) for i in range(self.batch_size)]

        # IoU 匹配predictor和label
        # 以Predictor为基准，每个Predictor都有且仅有一个需要负责的Target（前提是Predictor所在Grid Cell有Target中心位于此）
        # x, y, w, h, c
        match = []
        conf = []
        for i in range(self.batch_size):
            m, c = self.match_pred_target(input[i], target[i])
            match.append(m)
            conf.append(c)
        
        loss = torch.zeros([self.batch_size], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        for i in range(self.batch_size):
            loss[i], xy_loss[i], wh_loss[i], conf_loss[i], class_loss[i] = \
                self.compute_loss(input[i], target[i], match[i], conf[i])
        return torch.mean(loss), torch.mean(xy_loss), torch.mean(wh_loss), torch.mean(conf_loss), torch.mean(class_loss)

    def label_direct2grid(self, label):
        """
        :param label: dataset type: xmin, ymin, xmax, ymax, class
        :return: label: grid type, if the grid doesn't have object -> put None
        将label转换为c_x, c_y, w, h, conf再根据不同的grid cell分类，并转换成百分比形式
        若一个grid cell中没有label则都用None代替
        """
        # 输入是一张图里所有的target带bbox
        # 其实就是把dataset的原始形式转化为YOLO要用的形式

        output = [None for _ in range(self.S ** 2)]
        size = self.image_size // self.S  # pixel per grid

        bbox_num = label.size(0)
        label_c = torch.zeros_like(label)

        label_c[:, 0] = (label[:, 0] + label[:, 2]) / 2 # cx
        label_c[:, 1] = (label[:, 1] + label[:, 3]) / 2 # cy
        label_c[:, 2] = abs(label[:, 0] - label[:, 2])  # w
        label_c[:, 3] = abs(label[:, 1] - label[:, 3])  # h
        label_c[:, 4] = label[:, 4]                     # class

        # this part calculates in which grid the target belongs (cx, cy is in)
        idx_x = [int(label_c[i][0]) // size for i in range(bbox_num)]
        idx_y = [int(label_c[i][1]) // size for i in range(bbox_num)]

        label_c[:, 0] = torch.div(torch.fmod(label_c[:, 0], size), size) # cx position in that grid
        label_c[:, 1] = torch.div(torch.fmod(label_c[:, 1], size), size) # cy position in that grid
        label_c[:, 2] = torch.div(label_c[:, 2], self.image_size) # w: % form
        label_c[:, 3] = torch.div(label_c[:, 3], self.image_size) # h: % form

        for i in range(bbox_num):
            idx = idx_y[i] * self.S + idx_x[i]
            if output[idx] is None:
                output[idx] = torch.unsqueeze(label_c[i], dim=0) # the class should belong to center grid
            else:
                output[idx] = torch.cat([output[idx], torch.unsqueeze(label_c[i], dim=0)], dim=0)
        return output

    def match_pred_target(self, input, target):
        """
        :param input: (yolo net output)
                      tensor[s, s, b*5 + n_class] bbox: b * (c_x, c_y, w, h, obj_conf), class1_p, class2_p.. %
        :param target: (processed) tensor[n_bbox] bbox: x_min, ymin, xmax, ymax, class
        """
        # 这个函数的输入是某张图片的yolo输出(input)，某张图片的所有target(target)
        match = []
        conf = []
        with torch.no_grad():
            input_bbox = input[:, :self.B * 5].reshape(-1, self.B, 5)
            ious = [boxutils.match_get_iou(input_bbox[i], target[i], self.S, i)
                    for i in range(self.S ** 2)]
            for iou in ious:
                if iou is None:
                    match.append(None)
                    conf.append(None)
                else:
                    keep = np.ones([len(iou[0])], dtype=bool)
                    # 创建一个布尔类型的数组keep，用来表示哪些目标框已经被匹配过。
                    m = []
                    c = []
                    # 分别用来存储匹配的目标框索引和对应的IoU
                    for i in range(self.B): # 遍历每个预测框
                        if np.any(keep) == False: # 如果所有目标框都已经被匹配，则跳出循环
                            break
                        idx = np.argmax(iou[i][keep]) # 找到当前预测框和哪个目标框有最大的IoU
                        np_max = np.max(iou[i][keep]) # 获取最大的IoU值
                        m.append(np.argwhere(iou[i] == np_max).tolist()[0][0])
                        # 将匹配的目标框索引添加到列表中
                        c.append(np.max(iou[i][keep]))
                        # 将对应的IoU添加到列表中
                        keep[idx] = 0
                        # 标记该目标框已经被匹配
                    match.append(m)
                    conf.append(c)
        return match, conf

    def compute_loss(self, input, target, match, conf):
        # 计算loss，严格按照论文
        ce_loss = nn.CrossEntropyLoss()

        input_bbox  = input[:, :self.B * 5].reshape(-1, self.B, 5)
        input_class = input[:, self.B * 5:].reshape(-1, self.num_class)

        input_bbox = torch.sigmoid(input_bbox)
        loss = torch.zeros([self.S ** 2], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        # 不同grid cell分别计算再求和
        for i in range(self.S ** 2):
            # 0 xy_loss, 1 wh_loss, 2 conf_loss, 3 class_loss
            l = torch.zeros([4], dtype=torch.float, device=self.device)
            # Neg
            if target[i] is None:
                # λ_noobj = 0.5
                obj_conf_target = torch.zeros([self.B], dtype=torch.float, device=self.device)
                l[2] = torch.sum(torch.mul(0.5, torch.pow(input_bbox[i, :, 4] - obj_conf_target, 2)))
            else:
                # λ_coord = 5
                l[0] = torch.mul(5, torch.sum(torch.pow(input_bbox[i, :, 0] - target[i][match[i], 0], 2) +
                                              torch.pow(input_bbox[i, :, 1] - target[i][match[i], 1], 2)))

                l[1] = torch.mul(5, torch.sum(torch.pow(torch.sqrt(input_bbox[i, :, 2]) -
                                                        torch.sqrt(target[i][match[i], 2]), 2) +
                                              torch.pow(torch.sqrt(input_bbox[i, :, 3]) -
                                                        torch.sqrt(target[i][match[i], 3]), 2)))
                obj_conf_target = torch.tensor(conf[i], dtype=torch.float, device=self.device)
                l[2] = torch.sum(torch.pow(input_bbox[i, :, 4] - obj_conf_target, 2))

                l[3] = ce_loss(input_class[i].unsqueeze(dim=0).repeat(target[i].size(0), 1),
                               target[i][:, 4].long())
            loss[i]       = torch.sum(l)
            xy_loss[i]    = torch.sum(l[0])
            wh_loss[i]    = torch.sum(l[1])
            conf_loss[i]  = torch.sum(l[2])
            class_loss[i] = torch.sum(l[3])
        return torch.sum(loss), torch.sum(xy_loss), torch.sum(wh_loss), torch.sum(conf_loss), torch.sum(class_loss)