import torch.nn as nn
import torch


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        in_channels_to_be_modified = in_channels

        list_of_layers = []

        for _ in range(num_convs):
            list_of_layers.append(
                nn.Conv2d(
                    in_channels_to_be_modified,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )
            list_of_layers.append(nn.BatchNorm2d(out_channels))
            list_of_layers.append(nn.ReLU(inplace=True))

            in_channels_to_be_modified = out_channels

        list_of_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.sequential = nn.Sequential(*list_of_layers)


    def forward(self, x):
        return self.sequential(x)


class VGGNet(nn.Module):

    def __init__(self, num_classes=18):
        super(VGGNet, self).__init__()

        list_of_layers = []

        list_of_layers.extend([
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ])

        list_of_layers.append(VGGBlock(64, 128))
        list_of_layers.append(VGGBlock(128, 256))

        self.sequential = nn.Sequential(*list_of_layers)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_transformation = nn.Linear(256, num_classes)


    def forward(self, x):
        
        intermediate = self.sequential(x)
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, 1)
        return self.linear_transformation(intermediate)