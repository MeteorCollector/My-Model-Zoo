# SSD

SSD是两种单步目标识别网络之一。可以参考[之前的blog](https://meteorcollector.github.io/2024/03/cv-paper-2/)

[code reference](https://zhuanlan.zhihu.com/p/79854543)

[code reference](https://github.com/amdegroot/ssd.pytorch/tree/master)

主要是按照ssd.pytorch的源码进行实现。由于只想关心模型的主干部分，原项目的辅助函数我就直接引用了。这个仓库的代码解析能找到很多，`issue` 区也是十分精彩，百家争鸣。

可以说原作者在竭尽所能地确保代码的优美性，对于每个神经网络模块，总是采用传入config列表然后识别列表进行模块的产生，使用了大量的迭代和分类讨论。但是这样可读性非常糟糕，再加上论文中的SSD是一个固定的模型，我不太想关注它的可拓展性，所以我干脆把他的这种代码全部改写成了直接列举的形式，希望更加直观。这样以后复习的时候也能一眼就看出它的结构。

SSD的 `Loss` 是 `MultiBoxLoss`。计算方式如下：

定义$x_{ij}^p$为“第$i$个default box和第$j$个ground truth box匹配且类别为$p$”的指示器变量（indicator）。loss function为：

$$L(x, c, l, g) = \frac{1}{N}(L_{conf}(x, c) + \alpha L_{loc}(x, l, g))$$

其中$L_{loc}$（定位损失）和之前讲Faster R-CNN时提到的bounding-box regression类似。定义：$l$为预测框，$g$为ground-truth框，用这两个参数产生smooth L1 loss。回归默认框($d$)的中心($cx, cy$)和宽度、高度($w, h$)。

$$L_{loc}(x, l, g) = \sum^N_{i \in Pos}\sum_{m \in \{cx, cy, w, g\}}x^k_{ij}\mathrm{smooth L1}(\^m_i - \hat{g}^m_j)$$

$$\hat{g}^{cx}_j = (g^{cx}_j - d^{cx}_i) / d^w_i\;\;\;\;\hat{g}^{cy}_j = (g^{cy}_j - d_i^{cy}) / d^h_i = $$

$$\hat{g}^h_j = \log(\frac{g^h_j}{d^h_i})\;\;\;\;\hat{g}^h_j = \log(\frac{g^h_j}{d^h_i})$$

置信度损失是在多类别置信度（$c$）上的softmax损失。

$$L_{conf}(x, c) = -\sum^N_{i\in Pos} x^p_{ij}\log(\hat{c}^p_i) - \sum_{i \in Neg}\log(\hat{c}_i^0)$$

$$\hat{c}^p_i = \frac{\exp(c_i^p)}{\sum_p \exp(c^p_i)}$$