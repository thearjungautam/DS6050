import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # TODO: Assign to an instance attribute called `convolution_1` a 3x3 2D convolution with
        # a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels,
        # a stride equal to the provided stride, padding of 1, and no bias.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `batch_normalization_1` 2D batch normalization.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `relu` ReLU. ReLU must be performed in place.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `convolution_2` a 3x3 2D convolution with
        # a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels,
        # a stride of 1, padding of 1, and no bias.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `batch_normalization_2` 2D batch normalization.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called shortcut an empty object of type Sequential.
        # If the provided stride is not equal to 1 or
        # the provided number of input channels does not equal the provided number of output channels,
        # reassign shortcut an object of type Sequential constructed with a 1x1 2D convolution and 2D batch normalization.
        # The convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels, stride equal to the provided stride,
        # and no bias.
        raise NotImplementedError


    def forward(self, x):
        # TODO: Assign to a local variable called intermediate
        # the output of passing the input through the first convolution.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the first batch normalization.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through ReLU.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the second convolution.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the second batch normalization.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of adding intermediate and the output of passing the provided input
        # through the shortcut.
        raise NotImplementedError

        # TODO: Return the output of passing intermediate through ReLU.
        raise NotImplementedError


class ResNet(nn.Module):
    def __init__(self, num_classes=18):
        super(ResNet, self).__init__()
        
        # TODO: Assign to an instance attribute called `convolution` an object of type Sequential constructed with
        # 7x7 2D convolution, 2D batch normalization, and ReLU.
        # The convolution must have 3 input channels, 64 output channels, stride of 2, padding of 3, and no bias.
        # ReLU must be performed in place.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `max_pooling` 2D max pooling with
        # a kernel size of 3, stride of 2, and padding of 1.
        raise NotImplementedError

        # TODO: Assign to a instance attribute called `residual_layer_1` the output of method `_make_layer`
        # with 64 input channels, 64 output channels, 2 blocks, and stride of 1.
        raise NotImplementedError

        # TODO: Assign to a instance attribute called `residual_layer_2` the output of method `_make_layer`
        # with 64 input channels, 128 output channels, 2 blocks, and stride of 2.
        raise NotImplementedError

        # TODO: Assign to a instance attribute called `residual_layer_3` the output of method `_make_layer`
        # with 128 input channels, 256 output channels, 2 blocks, and stride of 2.
        raise NotImplementedError

        # TODO: Assign to a instance attribute called `residual_layer_4` the output of method `_make_layer`
        # with 256 input channels, 512 output channels, 2 blocks, and stride of 2.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `average_pooling` 2D average pooling with height of 1 and width of 1.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `linear_transformation` a linear transformation
        # with 512 input features and a number of output features equal to the provided number of classes.
        raise NotImplementedError


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        # TODO: Create an empty list called `list_of_blocks`.
        raise NotImplementedError

        # TODO: Add to the list of blocks a basic block with
        # a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels,
        # and a stride equal to the provided stride.
        raise NotImplementedError

        # TODO: For each remaining block, add to the list of blocks a basic block with
        # a number of input channels equal to the provided number of output channels and
        # a number of output channels equal to the provided number of output channels.
        raise NotImplementedError

        # TODO: Return an object of type Sequential constructed with the blocks in the list of blocks.
        raise NotImplementedError


    def forward(self, x):        
        # TODO: Assign to a local variable called intermediate the output of passing the provided input through
        # the convolution.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through max pooling.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the first residual layer.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the second residual layer.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the third residual layer.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the fourth residual layer.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through average pooling.
        raise NotImplementedError

        # TODO: Flatten intermediate from start dimension 1 on.
        # The output tensor of flattening has shape (number of images in batch, 512).
        raise NotImplementedError

        # TODO: Return the output of passing intermediate through the linear transformation.
        # The output tensor of the linear transformation has shape (number of images in batch, 18).
        raise NotImplementedError