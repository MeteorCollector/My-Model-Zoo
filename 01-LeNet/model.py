from torch.nn import Module
from torch import nn


class MyLeNet(Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        # equivalent to Parent.__init__(self)

        # define all layers here
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # instead of sigmoid, we use a more modern approach here.
        self.relu1 = nn.ReLU()
        # we use max pooling here
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.relu3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        # in the paper, input x is 32 * 32
        # but MNIST is 28 * 28
        # therefore padding is needed since feature map here is 4 * 4
        self.conv5 = nn.Conv2d(16, 120, 4)
        self.relu5 = nn.ReLU()

        # this additional linear layer
        # is because of 256 batch size
        self.func0 = nn.Linear(256, 120)
        self.relu0 = nn.ReLU()
        self.func6 = nn.Linear(120, 84)
        self.relu6 = nn.ReLU()
        self.outnn = nn.Linear(84, 10)
        self.relu7 = nn.ReLU()

    def forward(self, x):
        # connect all layers here
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.pool4(y)
        # 16 conv kernels are applied to all channels here
        # instead of manual settings in original paper.
        # y = self.conv5(y)
        # y = self.relu5(y)

        y = y.view(y.shape[0], -1)
        y = self.func0(y)
        y = self.relu0(y)
        y = self.func6(y)
        y = self.relu6(y)
        y = self.outnn(y)
        y = self.relu7(y)
        return y
