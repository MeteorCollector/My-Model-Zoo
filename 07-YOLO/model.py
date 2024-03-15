from modules import ConvModule
import numpy as np
import torch
import torch.nn as nn
import torchvision

class MyYOLO(nn.Module):
    def __init__(self, grid, out_channel):
        """
        grid is the amount of YOLO grids per row/column.
        """

        super(MyYOLO, self).__init__()
        self.backbone = ConvModule
        # in referenced repo,
        # author used a pretrained res50 as backbone
        # we just observe the original configuration.
        self.grid = grid

        self.fc = nn.Sequential(
            nn.Linear(1024 * grid * grid, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, out_channel * grid * grid)
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = self.backbone(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        y = y.view(batch_size, self.grid ** 2, -1)
        return y


class yolo_loss:
    def __init__(self, device, grid, b, image_size, num_class):
        self.device = device
        self.grid   = grid
        self.b      = b
        self.image_size = image_size
        self.num_class  = num_class
        self.batch_size = 0