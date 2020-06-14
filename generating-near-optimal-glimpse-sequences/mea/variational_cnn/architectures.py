import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v2


class TwoClassifiers(nn.Module):
    def __init__(self, classifier, num_labels, num_classes):
        super().__init__()
        self.sigmoid = replace_with_sigmoid_classifier(classifier,
                                                       num_labels)
        self.softmax = replace_with_logsoftmax_classifier(classifier,
                                                          num_classes)
    def forward(self, x):
        return {"sigmoid": self.sigmoid(x),
                "softmax": self.softmax(x)}


def replace_with_sigmoid_classifier(classifier, num_labels):
    num_final_features = classifier.in_features
    classifier = nn.Sequential(
        nn.Linear(num_final_features, num_labels),
        nn.Sigmoid()
    )
    return classifier


def replace_with_logsoftmax_classifier(classifier, num_classes):
    num_final_features = classifier.in_features
    classifier = nn.Sequential(
        nn.Linear(num_final_features, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return classifier


def add_conv_channel(conv):
    assert conv.bias is None
    init_weights = conv.weight.data
    conv = nn.Conv2d(4, conv.out_channels,
                     kernel_size=conv.kernel_size,
                     stride=conv.stride,
                     padding=conv.padding,
                     bias=False)
    conv.weight.data[:, 0:3, :, :] = init_weights
    conv.weight.data[:, 3, :, :] = 0*conv.weight.data[:, 3, :, :]
    return conv


def get_densenet(num_labels=None, num_classes=None,
                 pretrained_net=None,):
    """
    if pretrained net is provided, uses this and just modifies last layer.
    otherwise takes one pretrained on imagenet and adds channel to initial conv as well.
    """
    assert not (num_labels is None and num_classes is None)
    if pretrained_net is None:
        net = models.densenet121(pretrained=True)
        net.features.conv0 = add_conv_channel(net.features.conv0)
    else:
        net = pretrained_net
    c = net.classifier
    if num_labels is not None and num_classes is not None:
        c = TwoClassifiers(c, num_labels, num_classes)
    elif num_labels is None and num_classes is not None:
        c = replace_with_logsoftmax_classifier(c, num_classes)
    else:
        c = replace_with_sigmoid_classifier(c, num_labels)
    net.classifier = c
    return net


def get_mobilenet(num_labels=None, num_classes=None,
                  pretrained_net=None,):
    """
    if pretrained net is provided, uses this and just modifies last layer.
    otherwise takes one pretrained on imagenet and adds channel to initial conv as well.
    """
    assert not (num_labels is None and num_classes is None)
    if pretrained_net is None:
        net = mobilenet_v2(pretrained=True)
        net.features[0][0] = add_conv_channel(net.features[0][0])
    else:
        net = pretrained_net
    c = net.classifier[1]
    if num_labels is not None and num_classes is not None:
        c = TwoClassifiers(c, num_labels, num_classes)
    elif num_labels is None and num_classes is not None:
        c = replace_with_logsoftmax_classifier(c, num_classes)
    else:
        c = replace_with_sigmoid_classifier(c, num_labels)
    net.classifier[1] = c
    return net
