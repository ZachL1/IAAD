import torchvision.models as models


def vgg16(pretrained=True, **kwargs):
    return models.vgg16(pretrained=pretrained).features

def vgg16_bn(pretrained=True, **kwargs):
    return models.vgg16_bn(pretrained=pretrained).features