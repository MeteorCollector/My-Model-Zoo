# VGG

## 介绍

VGG是Oxford的Visual Geometry Group的组提出的（大家应该能看出VGG名字的由来了）。该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

[text reference](https://zhuanlan.zhihu.com/p/41423739)

## 原理

VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。

简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

## 网络结构

### VGG16

```
input: 224 * 224 RGB

conv3-64
conv3-64
maxpool

conv3-128
conv3-128
maxpool

conv3-256
conv3-256
conv3-256
maxpool

conv3-512
conv3-512
conv3-512
maxpool

conv3-512
conv3-512
conv3-512
maxpool

FC-4096
FC-4096
FC-1000
softmax
```

### VGG19

```
input: 224 * 224 RGB

conv3-64
conv3-64
maxpool

conv3-128
conv3-128
maxpool

conv3-256
conv3-256
conv3-256
conv3-256
maxpool

conv3-512
conv3-512
conv3-512
conv3-512
maxpool

conv3-512
conv3-512
conv3-512
conv3-512
maxpool

FC-4096
FC-4096
FC-1000
softmax
```