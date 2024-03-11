# 一些转载的pytorch相关笔记

## module 类

[reference](https://blog.csdn.net/qq_27825451/article/details/90550890)

我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：

（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
    
（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。

## super函数

在Python中，我们通常使用 `super()` 函数来引用父类。基础的用法是带有两个参数：第一个是子类，第二个是对象，即 `super(subclass, instance)`。其用途是返回一个临时对象，该对象绑定到父类的方法上，而不是子类的方法。

```python
class Parent():
    def hello(self):
        print("Hello from Parent")

class Child(Parent):
    def hello(self):
        super(Child, self).hello()
        print("Hello from Child")

c = Child()
c.hello()
```

这段代码会输出：

```
Hello from Parent
Hello from Child
```

本质上，`super(Child, self).hello()` 就等于 `Parent.hello(self)`。使用 `super` 的好处是不必显式写出父类具体是哪一个，这样有利于后续维护与更新（比如改变了父类，但是这段代码不用改）。

## .shape

[reference](https://blog.csdn.net/giganticpower/article/details/108081652)

shape用法

在CNN中，我们在接入全连接层的时候，我们需要将提取出的特征图进行铺平，将特征图转换为一维向量。
这时候我们用到.view做一个resize的功能，用.shape来进行选择通道数。

.shape()和.size()用法有点类似。
.shape[0]和.size(0)都是提取一维参量。

比如，
CNN特征图feature最终输出为(50,16,4,4)(B,C,H,W)
那么.shape[0]和.size(0)就是提取50这个数据。

## .view

```python
a=torch.Tensor([[[1,2,3],[4,5,6]]])
b=torch.Tensor([1,2,3,4,5,6])

print(a.view(1,6))
print(b.view(1,6))
```

得到的结果都是 `tensor([[1., 2., 3., 4., 5., 6.]])`

```python
a=torch.Tensor([[[1,2,3],[4,5,6]]])
print(a.view(3,2))
```

得到结果：`tensor([[1., 2.],[3., 4.],[5., 6.]])`

参数中的 `-1` 就代表这个位置由其他位置的数字来推断，只要在不致歧义的情况的下，`view` 参数就可以推断出来，也就是人可以推断出形状的情况下，`view` 函数也可以推断出来。比如a tensor的数据个数是6个，如果`view（1，-1）`，我们就可以根据tensor的元素个数推断出-1代表6。

## General step of training

```python
for i, (image, label) in enumerate(train_loader):
    # 1. forward
    pred = model(image)
    loss = criterion(pred, label)

    # 2. backward
    loss.backward()

    # 3. update parameters of net
    optimizer.step() 
 
    # 4. reset gradient
    optimizer.zero_grad()
```