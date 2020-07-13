import math
import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PSPModule(nn.Module):
    def __init__(self, feat_dim, bins=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.reduction_dim = feat_dim // len(bins)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(feat_dim, size) for size in bins])

    def _make_stage(self, feat_dim, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(feat_dim, self.reduction_dim, kernel_size=1, bias=False)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [feats]
        for stage in self.stages:
            priors.append(F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(priors, 1)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), backend='resnet18'):
        super(PSPNet, self).__init__()
        if backend == 'resnet18':
            self.feats = ResNet(BasicBlock, [2, 2, 2, 2])
            feat_dim = 512
        else:
            raise NotImplementedError
        self.psp = PSPModule(feat_dim, bins)
        self.drop = nn.Dropout2d(p=0.15)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.final = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        f = self.feats(x)
        p = self.psp(f)
        p = self.up_1(p)
        p = self.drop(p)
        p = self.up_2(p)
        p = self.drop(p)
        p = self.up_3(p)
        return self.final(p)
