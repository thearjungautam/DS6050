import torch.nn as nn
import torch


class NiNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()

        list_of_layers = []

        list_of_layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        list_of_layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        list_of_layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.sequential = nn.Sequential(*list_of_layers)



    def forward(self, x):
        return self.sequential(x)


class NiN(nn.Module):
    def __init__(self, num_classes=18):
        super(NiN, self).__init__()

        list_of_layers = []

        list_of_layers.append(
            NiNBlock(3, 96, kernel_size=11, stride=4)
        )
        list_of_layers.append(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        list_of_layers.append(
            NiNBlock(96, 256, kernel_size=5, padding=2)
        )
        list_of_layers.append(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        list_of_layers.append(
            NiNBlock(256, 384, kernel_size=3, padding=1)
        )
        list_of_layers.append(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        list_of_layers.append(
            NiNBlock(384, num_classes, kernel_size=3, padding=1)
        )

        list_of_layers.append(
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.sequential = nn.Sequential(*list_of_layers)



    def forward(self, x):
        intermediate = self.sequential(x)
        return torch.flatten(intermediate, 1)