# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import math

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm


# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)  # split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2;  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10;  # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(
            x_normalized)  # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        C = nn.Conv2d(indim, outdim, 3, padding=padding)
        BN = nn.BatchNorm2d(outdim)
        relu = nn.ReLU(inplace=True)

        parametrized_layers = [C, BN, relu]
        if pool:
            pool = nn.MaxPool2d(2)
            parametrized_layers.append(pool)

        for layer in parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        self.layer1 = ConvBlock(3, 64, pool=True)
        self.layer2 = ConvBlock(64, 64, pool=True)
        self.layer3 = ConvBlock(64, 64, pool=True)
        self.layer4 = nn.Sequential(
            *[ConvBlock(64, 64, pool=(i == 0)) for i in range(depth - 3)])  # only pooling for fist 4 layers
        self.do_flatten = flatten
        if flatten:
            self.flatten = Flatten()
        self.fpn_sizes = [64, 64, 64, 64]
        self.final_feat_dim = 512  # if you use our method
        # self.final_feat_dim = 1600#if not

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = out4
        if self.do_flatten:
            out = self.flatten(out)
        return out2, out3, out4, out


class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.layer1 = ConvBlock(3, 64, pool=True, padding=0)
        self.layer2 = ConvBlock(64, 64, pool=True, padding=0)
        self.layer3 = ConvBlock(64, 64, pool=False, padding=1)
        self.layer4 = ConvBlock(64, 64, pool=False, padding=1)
        self.fpn_sizes = [64, 64, 64, 64]
        self.final_feat_dim = [512, 19, 19]  # 512 channels if you use our method
        # self.final_feat_dim = [64,19,19]#64 channels if not
        self.do_flatten = False

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out2, out3, out4, out4


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet(nn.Module):

    def __init__(self, channels, flatten):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.do_flatten = flatten
        if flatten:
            self.flatten = Flatten()
            self.final_feat_dim = 1920  # if use our method
        else:
            self.final_feat_dim = [1920, 5, 5]  # if use our method

        self.fpn_sizes = channels
        # self.final_feat_dim=12800#if not
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = x4
        if self.do_flatten:
            out = self.flatten(out)
        return x2, x3, x4, out


def Conv4():
    return ConvNet(4)


def Conv4NP():
    return ConvNetNopool(4)


def ResNet12(flatten=True):
    model = ResNet([64, 128, 256, 512], flatten)
    return model


if __name__ == '__main__':
    model = ResNet12()
    print(model)
    x = torch.randn(1, 3, 84, 84)
    out1, out2, out3, out4 = model(x)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
