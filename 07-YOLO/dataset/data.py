# 路径: ./dataset/data.py
from dataset.transform import *     # 导入我们重写的transform类

from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import json
import os


def get_file_name(root, layout_txt):
    with open(os.path.join(root, layout_txt)) as layout_txt:
        """
        .read()         读取文件中的数据，会得到一个str字符串
        .split('\n')    以\n回车符为分界将str字符串分割成list
        [:-1]           去除最后一个空字符串，文件末尾有\n，分割后会有空字符串所以要去除
        """
        file_name = layout_txt.read().split('\n')[:-1]
    return file_name


def xml2dict(xml):
    # 这里的xml2dict与上一个文件的xml2dict一致
    data = {c.tag: None for c in xml}
    for c in xml:
        def add(data, tag, text):
            if data[tag] is None:
                data[tag] = text
            elif isinstance(data[tag], list):
                data[tag].append(text)
            else:
                data[tag] = [data[tag], text]
            return data

        if len(c) == 0:
            data = add(data, c.tag, c.text)
        else:
            data = add(data, c.tag, xml2dict(c))
    return data


class VOC0712Dataset(Dataset):
    def __init__(self, root, class_path, transforms, mode, data_range=None, get_info=False):
        # label: xmin, ymin, xmax, ymax, class

        # 从json文件中获得class的信息
        with open(class_path, 'r') as f:
            json_str = f.read()
            self.classes = json.loads(json_str)

        """
        如果是train模式，那么root的输入将为一个list（长为2，分别为2007、2012两年的数据集根目录， main中的root0712是一个示例）。将两个root与train、val两种分割组合成四个layout_txt路径，这四个路径指向VOC07/12的所有可用训练数据。
        如果是test模式那么只有VOC2007的test分割可用。这里也转换为list形式，就可以同一两种模式的代码。
        """
        layout_txt = None
        if mode == 'train':
            root = [root[0], root[0], root[1], root[1]]
            layout_txt = [r'ImageSets\Main\train.txt', r'ImageSets\Main\val.txt',
                          r'ImageSets\Main\train.txt', r'ImageSets\Main\val.txt']
        elif mode == 'test':
            if not isinstance(root, list):
                root = [root]
            layout_txt = [r'ImageSets\Main\test.txt']
        assert layout_txt is not None, 'Unknown mode'


        self.transforms = transforms
        self.get_info = get_info    # get_info表示在getitem时是否需要获得图像的名称以及图像大小信息 bool

        # 由于有多root，所以image_list与annotation_list均存储了图像与xml文件的绝对路径
        self.image_list = []
        self.annotation_list = []
        for r, txt in zip(root, layout_txt):
            self.image_list += [os.path.join(r, 'JPEGImages', t + '.jpg') for t in get_file_name(r, txt)]
            self.annotation_list += [os.path.join(r, 'Annotations', t + '.xml') for t in get_file_name(r, txt)]

        # data_range是一个二元tuple，分别表示数据集需要取哪一段区间，训练时若使用全部的数据则无需传入data_range，默认None的取值是会选择所有的数据的
        if data_range is not None:
            self.image_list = self.image_list[data_range[0]: data_range[1]]
            self.annotation_list = self.annotation_list[data_range[0]: data_range[1]]

    def __len__(self):
        # 返回数据集长度
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        image_size = image.size
        label = self.label_process(self.annotation_list[idx])

        if self.transforms is not None:
            # 由于目标检测中image的变换如随机裁剪与Resize都会导致label的变化，所以需要重写transform，添加部分的label处理代码
            image, label = self.transforms(image, label)
        if self.get_info:
            return image, label, os.path.basename(self.image_list[idx]).split('.')[0], image_size
        else:
            return image, label

    def label_process(self, annotation):
        xml = ET.parse(os.path.join(annotation)).getroot()
        data = xml2dict(xml)['object']
        # 根据data的两种形式将其读取到label中，并将label转为numpy形式
        if isinstance(data, list):
            label = [[float(d['bndbox']['xmin']), float(d['bndbox']['ymin']),
                     float(d['bndbox']['xmax']), float(d['bndbox']['ymax']),
                     self.classes[d['name']]]
                     for d in data]
        else:
            label = [[float(data['bndbox']['xmin']), float(data['bndbox']['ymin']),
                     float(data['bndbox']['xmax']), float(data['bndbox']['ymax']),
                     self.classes[data['name']]]]
        label = np.array(label)
        return label


if __name__ == "__main__":
    from dataset.draw_bbox import draw

    root0712 = [r'F:\AI\Dataset\VOC2007\VOCdevkit\VOC2007', r'F:\AI\Dataset\VOC2012\VOCdevkit\VOC2012']

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        Resize(448)
    ])
    ds = VOC0712Dataset(root0712, 'classes.json', transforms, 'train', get_info=True)
    print(len(ds))
    for i, (image, label, image_name, image_size) in enumerate(ds):
        if i <= 1000:
            continue
        elif i >= 1010:
            break
        else:
            print(label.dtype)
            print(tuple(image.size()[1:]))
            draw(image, label, ds.classes)
    print('VOC2007Dataset')