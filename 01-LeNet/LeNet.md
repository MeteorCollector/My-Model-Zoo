# LeNet

识别手写数字，万物之源
[reference](https://cuijiahua.com/blog/2018/01/dl_3.html)

由于是第一次写，所以还是要参照别人的代码
[reference](https://github.com/ChawDoe/LeNet5-MNIST-PyTorch)

不过他的代码并不太忠于原论文，我做了一些修改

## 概述

LeNet-5是一个较简单的卷积神经网络。输入的二维图像，先经过两次卷积层到池化层，再经过全连接层，最后使用softmax分类作为输出层。

LeNet-5 这个网络虽然很小，但是它包含了深度学习的基本模块：卷积层，池化层，全连接层。是其他深度学习模型的基础。

LeNet-5共有7层，不包含输入，每层都包含可训练参数；每个层有多个Feature Map，每个FeatureMap通过一种卷积滤波器提取输入的一种特征，然后每个FeatureMap有多个神经元。

## 逐层解析

### INPUT层-输入层

首先是数据 INPUT 层，输入图像的尺寸统一归一化为32*32。

注意：本层不算LeNet-5的网络结构，传统上，不将输入层视为网络层次结构之一。

### C1层-卷积层

输入图片：$32 \times 32$

卷积核大小：$5 \times 5$

卷积核种类：$6$

输出featuremap大小：$28 \times 28$ $(32-5+1)=28$

神经元数量：$28 \times 28 \times 6$

可训练参数：$(5 \times 5 + 1)\times 6$(每个滤波器$5 \times 5=25$个unit参数和一个bias参数，一共6个滤波器)

连接数：$(5\times 5+1)\times 6\times 28\times 28=122304$

### S2层-池化层

输入：$28 \times 28$

采样区域：$2 \times 2$

采样方式：4个输入相加，乘以一个可训练参数，再加上一个可训练偏置。结果通过sigmoid。可以发现和现在的主流Pooling和激活函数设置是不一样的。

采样种类：$6$

输出featureMap大小：$14 \times 14$ $(28 / 2)$

神经元数量：$14 \times 14 \times 6$

可训练参数：$2 \times 6$ (6个通道分别的权重weight和偏置bias)

连接数：$(2 \times 2 + 1) \times 6 \times 14 \times 14$

### C3层-卷积层

卷积核大小：$5 \times 5$

卷积核种类：$16$

输出featureMap大小：$10 \times 10$ $(14 - 5 + 1) = 10$

注意这里有特殊的组合计算。

C3的前6个feature map与S2层相连的3个feature map相连接，后面6个feature map与S2层相连的4个feature map相连接，后面3个feature map与S2层部分不相连的4个feature map相连接，最后一个与S2层的所有feature map相连。卷积核大小依然为$5\times 5$，所以总共有$6\times (3\times 5\times 5+1)+6\times (4\times 5\times 5+1)+3\times (4\times 5\times 5+1)+1\times (6\times 5\times 5+1)=1516$个参数。而图像大小为10*10，所以共有151600个连接。

(实际上是采用池化层featureMap的channel的不同的组合模式)

论文中给出了两个如此设置的原因：1）减少参数，2）这种不对称的组合连接的方式有利于提取多种组合特征。

### S4层-池化层（下采样层）

输入：$10 \times 10$

采样区域：$2 \times 2$

采样方式：4个输入相加，乘以一个可训练参数，再加上一个可训练偏置。结果通过sigmoid。

采样种类：$16$

输出featureMap大小：$5 \times 5$ $(10 / 2)$

神经元数量：$5 \times 5 \times 16 = 400$

可训练参数：$2 \times 16$ (16个通道分别的权重weight和偏置bias)

连接数：$(2 \times 2 + 1) \times 6 \times 5 \times 5 = 2000$

### C5层-卷积层

输入：$5 \times 5$

卷积核大小：$5 \times 5$

卷积核种类：$120$

输出featuremap大小：$1 \times 1$ $(5-5+1)=1$

可训练参数：$(16 \times 5 \times 5 + 1)\times 120$

### F6层-全连接层

输入：c5 120维向量

计算方式：计算输入向量和权重向量之间的点积，再加上一个偏置，结果通过sigmoid函数输出。

可训练参数:84*(120+1)=10164

### Output层-全连接层

Output层也是全连接层，共有10个节点，分别代表数字0到9，且如果节点i的值为0，则网络识别的结果是数字i。采用的是径向基函数（RBF）的网络连接方式。假设x是上一层的输入，y是RBF的输出，则RBF输出的计算方式是：

$$y_i = \sum_i(x_j - w_{ij})^2$$