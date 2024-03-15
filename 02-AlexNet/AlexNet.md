# AlexNet

## 介绍

AlexNet是由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton在2012年ImageNet图像分类竞赛中提出的一种经典的卷积神经网络。当时，AlexNet在 ImageNet 大规模视觉识别竞赛中取得了优异的成绩，把深度学习模型在比赛中的正确率提升到一个前所未有的高度。因此，它的出现对深度学习发展具有里程碑式的意义。

[text reference](https://zhuanlan.zhihu.com/p/618545757)
[code reference](https://github.com/Lornatang/AlexNet-PyTorch)

## 网络结构

### Input

AlexNet输入为RGB三通道的224 × 224 × 3大小的图像（也可填充为227 × 227 × 3 ）

### Conv + Pooling

#### Conv1

input: $224 \times 224 \times 3$

kernel size: $11 \times 11 \times 3$

channel: $96$

stride: $4$

feature map: split $55 \times 55 \times 96$ into 2 $55 \times 55 \times 48$ submaps, connect to

#### ReLU1

#### Pool1

size: $3 \times 3$

stride: $2$

output: $2 \times 27 \times 27 \times 48$

#### LRN1

#### Conv2

kernel size: $5 \times 5 \times 48$

channel: $256$

#### ReLU2

#### Pool2

#### LRN2

#### Conv3

kernel size: $3 \times 3 \times 256$

channel: $384$

#### ReLU3

#### Conv4

kernel size: $3 \times 3 \times 192$

channel: $384$

#### ReLU4

#### Conv5

kernel size: $3 \times 3 \times 192$

channel: $256$

#### ReLU5

#### Pool5

size: $3 \times 3$

stride: $2$

### Full

#### F6

Get $4096$ channels of $1 \times 1$ feature map from pooling. Used 

#### ReLU6

#### Dropout

#### F7

#### ReLU7

#### Dropout

#### F8

#### Softmax

$1000$ dimension output.
