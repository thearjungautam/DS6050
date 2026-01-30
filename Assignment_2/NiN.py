import torch.nn as nn
import torch


class NiNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()

        # TODO: Create an empty list of layers.
        raise NotImplementedError

        # TODO: Add to the list of layers 2D convolution, 2D batch normalization, and ReLU.
        # The convolution must have the provided number of input channels,
        # number of output channels, kernel size, stride, and padding.
        # The convolution must have no bias.
        # Perform ReLU in place.
        raise NotImplementedError

        # TODO: Add to the list of layers 2D convolution, batch normalization, and ReLU.
        # The convolution must have a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels, a kernel size of 1, and no bias.
        # Perform ReLU in place.
        raise NotImplementedError

        # TODO: Add to the list of layers 2D convolution, batch normalization, and ReLU.
        # The convolution must have a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels, a kernel size of 1, and no bias.
        # Perform ReLU in place.
        raise NotImplementedError

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError


    def forward(self, x):
        # TODO: Return the output of passing the provided input to sequential.
        raise NotImplementedError


class NiN(nn.Module):
    def __init__(self, num_classes=18):
        super(NiN, self).__init__()

        # TODO: Create an empty list of layers.
        raise NotImplementedError

        # TODO: Add to the list of layers a NiN block with 3 input channels,
        # 96 output channels, a kernel size of 11, and a stride of 4.
        raise NotImplementedError

        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        raise NotImplementedError

        # TODO: Add to the list of layers a NiN block with 96 input channels,
        # 256 output channels, a kernel size of 5, and padding of 2.
        raise NotImplementedError
        
        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        raise NotImplementedError

        # TODO: Add to the list of layers a NiN block with 256 input channels,
        # 384 output channels, a kernel size of 3, and padding of 1.
        raise NotImplementedError
        
        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        raise NotImplementedError

        # TODO: Add to the list of layers a NiN block with 384 input channels,
        # a number of output channels equal to the provided number of classes, a kernel size of 3, and padding of 1.
        raise NotImplementedError

        # TODO: Add to the list of layers 2D average pooling with height of 1 and width of 1.
        raise NotImplementedError

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError


    def forward(self, x):
        # TODO: Assign to a local variable called intermediate
        # the output of passing the provided input to sequential.
        raise NotImplementedError

        # TODO: Return the output of flattening intermediate from start dimension 1 on.
        # The output tensor of flattening has shape (number of images in batch, number of classes).
        raise NotImplementedError