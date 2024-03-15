import torch
import numpy as np

def cxcywh2xyxy(bbox):
    """
    :param bbox: [bbox, bbox, ..] tensor c_x(%), c_y(%), w(%), h(%), c
    """
    bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox


def match_get_iou(bbox1, bbox2, s, idx):
    """
    :param bbox1: [bbox, bbox, ..] tensor c_x(%), c_y(%), w(%), h(%), c
    :return:
    """

    if bbox1 is None or bbox2 is None:
        return None

    bbox1 = np.array(bbox1.cpu())
    bbox2 = np.array(bbox2.cpu())

    # c_x, c_y转换为对整张图片的百分比
    bbox1[:, 0] = bbox1[:, 0] / s
    bbox1[:, 1] = bbox1[:, 1] / s
    bbox2[:, 0] = bbox2[:, 0] / s
    bbox2[:, 1] = bbox2[:, 1] / s

    # c_x, c_y加上grid cell左上角左边变成完整坐标
    grid_pos = [(j / s, i / s) for i in range(s) for j in range(s)]
    bbox1[:, 0] = bbox1[:, 0] + grid_pos[idx][0]
    bbox1[:, 1] = bbox1[:, 1] + grid_pos[idx][1]
    bbox2[:, 0] = bbox2[:, 0] + grid_pos[idx][0]
    bbox2[:, 1] = bbox2[:, 1] + grid_pos[idx][1]

    bbox1 = cxcywh2xyxy(bbox1)
    bbox2 = cxcywh2xyxy(bbox2)

    # %
    return get_iou(bbox1, bbox2)


def get_iou(bbox1, bbox2):
    """
    :param bbox1: [bbox, bbox, ..] tensor xmin ymin xmax ymax
    :param bbox2:
    :return: area:
    """

    s1 = abs(bbox1[:, 2] - bbox1[:, 0]) * abs(bbox1[:, 3] - bbox1[:, 1])
    s2 = abs(bbox2[:, 2] - bbox2[:, 0]) * abs(bbox2[:, 3] - bbox2[:, 1])

    ious = []
    for i in range(bbox1.shape[0]):
        xmin = np.maximum(bbox1[i, 0], bbox2[:, 0])
        ymin = np.maximum(bbox1[i, 1], bbox2[:, 1])
        xmax = np.minimum(bbox1[i, 2], bbox2[:, 2])
        ymax = np.minimum(bbox1[i, 3], bbox2[:, 3])

        in_w = np.maximum(xmax - xmin, 0)
        in_h = np.maximum(ymax - ymin, 0)

        in_s = in_w * in_h

        iou = in_s / (s1[i] + s2 - in_s)
        ious.append(iou)
    ious = np.array(ious)
    return ious


def nms(bbox, conf_th, iou_th):
    bbox = np.array(bbox.cpu())

    bbox[:, 4] = bbox[:, 4] * bbox[:, 5]

    bbox = bbox[bbox[:, 4] > conf_th]
    order = np.argsort(-bbox[:, 4])

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        iou = get_iou(np.array([bbox[i]]), bbox[order[1:]])[0]
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return bbox[keep]


# yolo后处理
def output_process(output, image_size, s, b, conf_th, iou_th):
    """
    输入是包含batch的模型输出
    :return output: list[], bbox: xmin, ymin, xmax, ymax, obj_conf, classes_conf, classes
    """
    batch_size = output.size(0)
    size = image_size // s

    output = torch.sigmoid(output)

    # Get Class
    # 将class conf依次添加到bbox中
    classes_conf, classes = torch.max(output[:, :, b * 5:], dim=2)
    classes = classes.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    classes_conf = classes_conf.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    bbox = output[:, :, :b * 5].reshape(batch_size, -1, b, 5)

    bbox = torch.cat([bbox, classes_conf, classes], dim=3)

    # To Direct
    # 百分比形式转直接表示
    bbox[:, :, :, [0, 1]] = bbox[:, :, :, [0, 1]] * size
    bbox[:, :, :, [2, 3]] = bbox[:, :, :, [2, 3]] * image_size

    # 添加grid cell坐标
    grid_pos = [(j * image_size // s, i * image_size // s) for i in range(s) for j in range(s)]

    def to_direct(bbox):
        for i in range(s ** 2):
            bbox[i, :, 0] = bbox[i, :, 0] + grid_pos[i][0]
            bbox[i, :, 1] = bbox[i, :, 1] + grid_pos[i][1]
        return bbox

    bbox_direct = torch.stack([to_direct(b) for b in bbox])
    bbox_direct = bbox_direct.reshape(batch_size, -1, 7)

    # cxcywh to xyxy
    bbox_direct[:, :, 0] = bbox_direct[:, :, 0] - bbox_direct[:, :, 2] / 2
    bbox_direct[:, :, 1] = bbox_direct[:, :, 1] - bbox_direct[:, :, 3] / 2
    bbox_direct[:, :, 2] = bbox_direct[:, :, 0] + bbox_direct[:, :, 2]
    bbox_direct[:, :, 3] = bbox_direct[:, :, 1] + bbox_direct[:, :, 3]

    bbox_direct[:, :, 0] = torch.maximum(bbox_direct[:, :, 0], torch.zeros(1))
    bbox_direct[:, :, 1] = torch.maximum(bbox_direct[:, :, 1], torch.zeros(1))
    bbox_direct[:, :, 2] = torch.minimum(bbox_direct[:, :, 2], torch.tensor([image_size]))
    bbox_direct[:, :, 3] = torch.minimum(bbox_direct[:, :, 3], torch.tensor([image_size]))

    # 整合不同batch中的bbox
    bbox = [torch.tensor(nms(b, conf_th, iou_th)) for b in bbox_direct]
    bbox = torch.stack(bbox)
    return bbox