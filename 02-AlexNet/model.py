from torch.nn import Module
from torch import nn
import torch

class MyAlexNet(Module):
    def __init__(self, config):
        super(MyAlexNet, self).__init__()
        
        # we have to config class number
        self.__config = config

        # this time we define in modules.
        self.conv_module = nn.Sequential(
            # 1
            nn.Conv2d(3, 96, 11, stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            # 2
            nn.Conv2d(96, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            # 3
            nn.Conv2d(256, 384, 3),
            nn.ReLU(),

            # 4
            nn.Conv2d(384, 384, 3),
            nn.ReLU(),

            # 5
            nn.Conv2d(384, 256, 3),
            nn.ReLU(),
        )

        # adaption layer, convert data to 6 * 6
        self.adapt = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.__config['num_classes']),
        )

    def forward(self, x):
        
        y = self.conv_module(x)
        # print(y.shape)
        y = self.adapt(y)
        # print(y.shape)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        
        return y
