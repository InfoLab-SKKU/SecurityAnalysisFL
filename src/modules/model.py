from torch import nn, optim
from torchvision import models
import torch


def mobilenet_v2(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def resnet101(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model = torch.nn.Sequential(
        *(list(model.children())[:-1]), torch.nn.Flatten(), nn.Linear(2048, num_classes)
    )
    model.hidden_dimension = 2048
    return model


def resnet18(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.hidden_dimension = model.fc.in_features
    return model


def vgg16(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.hidden_dimension = 4096
    return model


def efficientnet_b0(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.hidden_dimension = model.classifier[1].in_features
    return model


def densenet121(**kwargs):
    num_classes = kwargs.get('num_classes', 8)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.hidden_dimension = model.classifier.in_features
    return model


class ModelFactory:
    @staticmethod
    def create_model(config):
        model_name = config.model.name
        if model_name == 'mobilenet_v2':
            return mobilenet_v2(**config.model)
        if model_name == 'resnet101':
            return resnet101(**config.model)
        if model_name == 'resnet18':
            return resnet18(**config.model)
        if model_name == 'vgg16':
            return vgg16(**config.model)
        if model_name == 'efficientnet_b0':
            return efficientnet_b0(**config.model)
        if model_name == 'densenet121':
            return densenet121(**config.model)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
