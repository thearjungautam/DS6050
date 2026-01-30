import torch.nn as nn
import torch


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # TODO: Assign to an instance attribute called `depthwise_convolution` an object of type Sequential
        # constructed with 3x3 2D convolution, batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of input channels,
        # stride equal to the provided stride, a number of groups equal to the provided number of input channels,
        # padding of 1, and no bias.
        # Perform ReLU6 in place.
        raise NotImplementedError
        
        # TODO: Assign to an instance attribute called `pointwise_convolution` an object of type Sequential
        # constructed with 1x1 2D convolution, batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels, stride of 1, and no bias.
        # Perform ReLU6 in place.
        raise NotImplementedError


    def forward(self, x):

        # TODO: Assign to a local variable called intermediate the output of passing the input through depthwise convolution.
        raise NotImplementedError

        # TODO: Return the output of passing intermediate through pointwise convolution.
        raise NotImplementedError


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()

        # TODO: Assign to an instance attribute called `residual_will_be_used` an indicator
        # that stride is 1 and the number of input channels equals the number of output channels.
        raise NotImplementedError

        # TODO: Assign to a local variable called `hidden_dim`
        # the product of the provided number of input channels and the provided expansion ratio.
        raise NotImplementedError

        # TODO: Create an empty list called `list_of_layers`.
        raise NotImplementedError

        # TODO: If the provided expansion ratio is not equal to 1,
        #     add to the list of layers 1x1 2D convolution, 2D batch normalization, and ReLU6.
        #     The convolution must have a number of input channels equal to the provided number of input channels,
        #     a number of output channels equal to the hidden dimension, and no bias.
        #     Add to the list of layers batch normalization.
        #     Perform ReLU6 in place.
        raise NotImplementedError

        # TODO: Add to the list of layers 3x3 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the hidden dimension,
        # a number of output channels equal to the hidden dimension, stride equal to the provided stride,
        # a number of groups equal to the hidden dimension, padding of 1, and no bias.
        # Perform ReLU6 in place.
        raise NotImplementedError

        # TODO: Add to the list of layers 1x1 2D convolution and 2D batch normalization.
        # The convolution must have a number of input channels equal to the hidden dimension,
        # a number of output channels equal to the provided number of output channels, and no bias.
        raise NotImplementedError

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError


    def forward(self, x):

        # TODO: If residual will be used, return the output of adding the provided input and
        # the output of passing the provided input through sequential.
        # Otherwise, return the output of passing the provided input through sequential.
        raise NotImplementedError


class MobileNet(nn.Module):
    def __init__(self, num_classes=18, width_mult=1.0, dropout_prob=0.2):
        super(MobileNet, self).__init__()

        # TODO: Define a local variable called `number_of_output_channels_in_initial_convolution`
        # equal to the product of 32 and the provided multiplier.
        # Cast the product to an integer.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `initial_convolution` an object of type Sequential
        # constructed with a 3x3 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution must have a number of input channels of 3,
        # a number of output channels equal to the defined number of output channels in the initial convolution,
        # stride of 2, padding of 1, and no bias.
        # ReLU6 should be performed in place.
        raise NotImplementedError

        # TODO: Create an empty list of layers called `list_of_layers`.
        raise NotImplementedError

        configuration = [
            (32, 16, 1, 1),
            (16, 24, 2, 6),
            (24, 24, 1, 6),
            (24, 32, 2, 6),
            (32, 32, 1, 6),
            (32, 32, 1, 6),
            (32, 64, 2, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 160, 2, 6),
            (160, 160, 1, 6),
            (160, 160, 1, 6),
            (160, 320, 1, 6)
        ]

        # TODO: For number of input channels, number of output channels, stride, and expansion ratio in configuration,
        #     define a local variable called `scaled_number_of_input_channels` that is
        #     the product of the number of input channels and the provided multiplier.
        #     Cast the product to an integer.
        #     Define a local variable called `scaled_number_of_output_channels` that is
        #     the product of the number of output channels and the provided multiplier.
        #     Cast the product to an integer.
        #     Add to the list of layers an object of type `InvertedResidual` with
        #     a number of input channels equal to the scaled number of input channels,
        #     a number of output channels equal to the scaled number of output channels,
        #     the appropriate stride, and the appropriate expansion ratio.
        raise NotImplementedError

        # TODO: Add layers in the list of layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        raise NotImplementedError

        # TODO: Define a local variable called `number_of_output_channels_in_final_convolution`
        # equal to the product of 1280 and the provided multiplier if the multiplier is greater than 1.0 and 1280 otherwise.
        # Cast the product to an integer.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `final_convolution` an object of type Sequential
        # constructed with a 1x1 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution must have a number of input channels equal to the product of 320 and the provided multiplier,
        # a number of output channels equal to the defined number of output channels in the final convolution, and no bias.
        # Cast the product to an integer.
        # ReLU6 should be performed in place.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `average_pooling` 2D average pooling with height of 1 and width of 1.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `dropout` dropout with the provided probability.
        raise NotImplementedError

        # TODO: Assign to an instance attribute called `linear_transformation` a linear transformation
        # with a number of input features equal to the number of output channels in the final convolution and
        # a number of output features equal to the provided number of classes.
        raise NotImplementedError

        # TODO: Call method `initialize_weights`.
        raise NotImplementedError


    def initialize_weights(self):

        # TODO: For each module of this neural network,
        #     if the module is a 2D convolution,
        #         fill the module weight with values using a Kaiming normal distribution.
        #         if the module bias exists,
        #             fill the module bias with 0.
        #     otherwise, if the module is 2D batch normalization,
        #         fill the module weight with 1.
        #         Fill the module bias with 0.
        #     otherwise, if the module is a linear transformation,
        #         fill the module weight with values drawn from the standard normal distribution.
        #         Fill the module bias with 0.
        raise NotImplementedError


    def forward(self, x):

        # TODO: Assign to a local variable called intermediate the output of
        # passing the provided input through the initial convolution.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the sequential.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through the final convolution.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through average pooling.
        raise NotImplementedError

        # TODO: Flatten intermediate from start dimension 1 on.
        raise NotImplementedError

        # TODO: Assign to intermediate the output of passing intermediate through dropout.
        raise NotImplementedError

        # TODO: Return the output of passing intermediate through the linear transformation.
        raise NotImplementedError