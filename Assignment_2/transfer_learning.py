from typing import cast
from torchvision import models
import torch.nn as nn


def get_pretrained_model(model_name='resnet18', num_classes=18, feature_extract=True):
    
    # TODO: Assign to local variable model None.
    raise NotImplementedError

    # TODO: If model name is "resnet18",
    #    assign to model the output of function `resnet18` in module models.
    #    Function `resnet18` must receive for its parameter called weights
    #    the object DEFAULT from class `ResNet18_Weights` in module models.
    #    If parameters should be frozen,
    #        for each parameter in the parameters of the model,
    #            freeze that parameter.
    #    Redefine the linear transformation of the model as a linear transformation with
    #    a number of input features equal to the number of input features of the existing linear transformation and
    #    a number of output features equal to the provided number of classes.
    raise NotImplementedError
    
    # TODO: If model name is "vgg16",
    #    assign to model the output of function `vgg16` in module models.
    #    Function `vgg16` must receive for its parameter called weights
    #    the object DEFAULT from class `VGG16_Weights` in module models.
    #    If parameters should be frozen,
    #        for each parameter in the parameters of the features of the model,
    #            freeze that parameter.
    #        For each parameter in the parameters of the classifier of the model,
    #            freeze that parameter.
    #    Assign to a local variable called `layer_6` the output of casting to a linear transformation
    #    the layer with index 6 of the classifier of the model.
    #    Redefine the layer with index 6 of the classifier of the model as a linear transformation with
    #    a number of input features equal to the number of input features of `layer_6` and
    #    a number of output features equal to the provided number of classes.
    raise NotImplementedError

    # TODO: If model name is "mobilenet_v2",
    #    assign to model the output of function `mobilenet_v2` in module models.
    #    Function `mobilenet_v2` must receive for its parameter called weights
    #    the object DEFAULT from class `MobileNet_V2_Weights` in module models.
    #    If parameters should be frozen,
    #        for each parameter in the parameters of the model,
    #            freeze that parameter.
    #    Assign to a local variable called `layer_1` the output of casting to a linear transformation
    #    the layer with index 1 of the classifier of the model.
    #    Redefine the layer with index 1 of the classifier of the model as a linear transformation with
    #    a number of input features equal to the number of input features of `layer_1` and
    #    a number of output features equal to the provided number of classes.
    raise NotImplementedError

    # TODO: Otherwise,
    #    raise a value error with a message of "Model {model name} is unsupported.".
    raise NotImplementedError
    
    # TODO: Return the model.
    raise NotImplementedError