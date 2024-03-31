import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, depth=6, lr=0.001, channels=64):
        super(Model, self).__init__()
        self.depth = depth
        self.lr = lr

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1, channels, 3, padding=1))
        self.conv.extend([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(self.depth)])
        self.conv.append(nn.Conv2d(channels, 3, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(channels, channels) for _ in range(self.depth)])
        # initialize the weights of the Batch normalization layers
        for i in range(self.depth):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(channels))

    def forward(self, x):
        D = self.depth
        h = F.relu(self.conv[0](x))
        h_buff = []
        idx_buff = []
        shape_buff = []

        # Downsampling path
        for i in range(D // 2):
            shape_buff.append(h.shape)
            h, idx = F.max_pool2d(self.bn[i](F.relu(self.conv[i + 1](h))),
                                    kernel_size=2, stride=2, return_indices=True)
            h_buff.append(h)
            idx_buff.append(idx)

        # Bottleneck without pooling
        h = F.relu(self.bn[D // 2](self.conv[D // 2 + 1](h)))

        # Upsampling path
        for i in range(D // 2 + 1, D):
            j = D - 1 - i  # Correct the index here
            h = F.max_unpool2d(self.bn[i](F.relu(self.conv[i + 1]((h + h_buff[j]) / np.sqrt(2)))),
                                idx_buff[j], kernel_size=2, stride=2, output_size=shape_buff[j])

        # Final convolution layer
        y = self.conv[D + 1](h) + x
        return y
    

def createUDnCNN(depth=5, learning_rate=0.001):
    class UDnCNN(nn.Module):

        def __init__(self, D, C=64):
            super(UDnCNN, self).__init__()
            self.D = D

            # convolution layers
            self.conv = nn.ModuleList()
            self.conv.append(nn.Conv2d(1, C, 3, padding=1))
            self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
            self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
            # apply He's initialization
            for i in range(len(self.conv[:-1])):
                nn.init.kaiming_normal_(
                    self.conv[i].weight.data, nonlinearity='relu')

            # batch normalization
            self.bn = nn.ModuleList()
            self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
            # initialize the weights of the Batch normalization layers
            for i in range(D):
                nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

        def forward(self, x):
            D = self.D
            h = F.relu(self.conv[0](x))
            h_buff = []
            idx_buff = []
            shape_buff = []

            # Downsampling path
            for i in range(D // 2):
                shape_buff.append(h.shape)
                h, idx = F.max_pool2d(self.bn[i](F.relu(self.conv[i + 1](h))),
                                      kernel_size=2, stride=2, return_indices=True)
                h_buff.append(h)
                idx_buff.append(idx)

            # Bottleneck without pooling
            h = F.relu(self.bn[D // 2](self.conv[D // 2 + 1](h)))

            # Upsampling path
            for i in range(D // 2 + 1, D):
                j = D - 1 - i  # Correct the index here
                h = F.max_unpool2d(self.bn[i](F.relu(self.conv[i + 1]((h + h_buff[j]) / np.sqrt(2)))),
                                   idx_buff[j], kernel_size=2, stride=2, output_size=shape_buff[j])

            # Final convolution layer
            y = self.conv[D + 1](h) + x
            return y

    net = UDnCNN(D=depth)

    lossfun = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    return net, lossfun, optimizer

