# 路径: ./dataset/find_classes.py
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import os


def xml2dict(xml):
    """
    使用递归读取xml文件
    若c指向的元素是<name>person</name>，那么c.tag是name，c.text则是person
    """
    # data初始化时就已经将所有子元素的tag定义为key
    data = {c.tag: None for c in xml}
    for c in xml:
        # add函数用于将tag与text添加到data中
        def add(data, tag, text):
            if data[tag] is None:
                # data中该tag为空则直接添加text
                data[tag] = text
            elif isinstance(data[tag], list):
                # data中该tag为不为空且已经创建了list则append(text)
                data[tag].append(text)
            else:
                # data中该tag不为空但是没有创建list，需要先创建list
                data[tag] = [data[tag], text]
            return data

        if len(c) == 0:
            # len(c)表示c的子元素个数，若为0则表示c是叶元素，没有子元素
            data = add(data, c.tag, c.text)
        else:
            data = add(data, c.tag, xml2dict(c))
    return data


json_path = './classes.json'    # json保存到的地址

root = r'F:\AI\Dataset\VOC2012\VOCdevkit\VOC2012'   # 数据集root（VOC2007与VOC2012的class信息一致）
# 获取所有xml注释地址
annotation_root = os.path.join(root, 'Annotations')     
annotation_list = os.listdir(annotation_root)
annotation_list = [os.path.join(annotation_root, a) for a in annotation_list]

s = set()
for annotation in tqdm(annotation_list):
    xml = ET.parse(os.path.join(annotation)).getroot()
    data = xml2dict(xml)['object']
    if isinstance(data, list):
        # 有多个object
        for d in data:
            s.add(d['name'])
    else:
        # 仅有一个object
        s.add(data['name'])

s = list(s)
s.sort()

# 以class名称为key可以便于xml2label的转换
data = {value: i for i, value in enumerate(s)}
json_str = json.dumps(data)

with open(json_path, 'w') as f:
    f.write(json_str)