import torch.nn as nn
import torch


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        in_channels_to_be_modified = in_channels

        # TODO: Create an empty list of layers.
        raise NotImplementedError
    
        # TODO: Loop through the provided number of convolutions to add to the list of layers
        # 3x3 2D convolutions, 2D batch normalizations, and ReLUs.
        # The first convolution must have a number of input channels equal to the number of input channels to be modified and
        # a number of output channels equal to the provided number of output channels.
        # The remaining convolutions must have a number of input channels equal to the provided number of output channels and
        # a number of output channels equal to the provided number of output channels.
        # Each convolution must have padding of 1 so that the heights of the input and output tensors are the same
        # and the widths of the input and output tensors are the same.
        # Each convolution must have no bias.
        # Perform ReLU in place.
        raise NotImplementedError

        # TODO: Add to the list of layers 2x2 2D max pooling.
        # The output tensor of max pooling has shape
        # (number of images in batch, number of output channels, floor(height of input tensor / 2), floor(width of input tensor) / 2)).
        raise NotImplementedError

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError


    def forward(self, x):
        # TODO: Return the output of passing the provided input to sequential.
        raise NotImplementedError


class VGGNet(nn.Module):

    def __init__(self, num_classes=18):
        super(VGGNet, self).__init__()

        # TODO: Create an empty list of layers.
        raise NotImplementedError

        # TODO: Add to the list of layers 3x3 2D convolution, 2D batch normalization, and ReLU.
        # The convolution must have 3 input channels, 64 output channels, padding of 1, and no bias.
        # Perform ReLU in place.
        # The input tensor of this neural network has shape (number of images in batch, 3, height of image, width of image).
        # The output tensor of convolution, batch normalization, and ReLU has shape
        # (number of images in batch, 64, height of image, width of image).
        raise NotImplementedError

        # TODO: Add to the list of layers an object of type VGGBlock with 64 input channels and 128 output channels.
        # The output tensor of the first VGG block has shape
        # (number of images in batch, 128, floor(height of image / 2), floor(width of image) / 2)).
        raise NotImplementedError

        # TODO: Add to the list of layers an object of type VGGBlock with 128 input channels and 256 output channels.
        # The output tensor of the second VGG block has shape
        # (number of images in batch, 256, floor(height of input tensor / 2), floor(width of input tensor / 2)).
        raise NotImplementedError

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `average_pooling` 2D average pooling with height of 1 and width of 1.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `linear_transformation` a linear transformation
        # with 256 input features and a number of output features equal to the provided number of classes.
        raise NotImplementedError


    def forward(self, x):
        # The input tensor of this neural network has shape (number of images in batch, 3, height of image, width of image).

        # TODO: Assign to a local variable called intermediate
        # the output of passing the provided input to the object of type sequential of this instance.
        # The output tensor has shape (number of images in batch, 256, floor(height of image / 4), floor(width of image / 4)).
        raise NotImplementedError

        # TODO: Assign to intermediate to output of passing intermediate to the average pooling of this instance.
        # The output tensor of average pooling has shape (number of images in batch, 256, 1, 1).
        raise NotImplementedError

        # TODO: Flatten intermediate from start dimension 1 on.
        # The output tensor of flattening has shape (number of images in batch, 256).
        raise NotImplementedError

        # TODO: Return the output of passing intermediate to the linear transformation of this instance.
        # The output tensor of the linear transformation has shape (number of images in batch, 18).
        raise NotImplementedError