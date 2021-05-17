from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torchvision import models
#from torchsummary import summary
##vgg19 = models.vgg19(pretrained=True)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def vgg_Net(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class FFSTP(nn.Module):
    def __init__(self,x_dim=3, hid_dim=64, z_dim=64):
        super(FFSTP, self).__init__()

        self.conv1 = nn.Conv2d(1, 3 ,3, padding=1)

        self.LinearTransform2 = nn.Linear(1076,500)


        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7,padding=1),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True),
            nn.Conv2d(8, 1, kernel_size=5,padding=1),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(True)
        )
        self.LinearTransform1 = nn.Sequential(
            nn.Linear(1568, 1000),#288=size of hog feature
            nn.Linear(1000, 500)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear( 196 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        #self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 196* 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


    def forward_2(self,x):
        x = self.LinearTransform1(x)
        return x.view(x.size(0), -1)

    def forward_1(self, x):
        x = self.stn(x)
        x = self.conv1(x)
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def forward(self, x1, x2):
        out1 = self.forward_1(x1)
        out2 = self.forward_2(x2)
        x=torch.cat([out1,out2 ], dim=1)
        x=self.LinearTransform2(x)
        #print(x.view(x.size(0), -1))
        return x.view(x.size(0), -1)