import torch.nn as nn
#import qan
import torch
import torch.nn.init
import math
import torch.nn.functional as F

use_relu = False
use_bn = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    global use_bn
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=not use_bn)

def calculate_scale(data):
    if data.dim() == 2:
        scale = math.sqrt(3 / data.size(1))
    else:
        scale = math.sqrt(3 /
        (data.size(1) *
        data.size(2) *
        data.size(3)))
    return scale

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        global use_bn
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)

        torch.nn.init.normal_(self.conv1.weight.data, 0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.zero_()
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        if use_relu:
            self.relu1 = nn.ReLU(inplace = True)
        else:
            self.relu1 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)

        torch.nn.init.normal_(self.conv2.weight.data, 0, 0.01)
        if self.conv2.bias is not None:
            self.conv2.bias.data.zero_()
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if use_relu:
            self.relu2 = nn.ReLU(inplace = True)
        else:
            self.relu2 = nn.PReLU(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class ResNetCaffe(nn.Module):

    def __init__(self, layers, block=None, k = 1, use_relu_ = False, use_bn_ = True, pretrained=False):
        global use_relu
        use_relu = use_relu_
        global use_bn
        use_bn = use_bn_
        self.use_bn = use_bn
        self.inplanes = round(32 * k)
        super(ResNetCaffe, self).__init__()
        self.conv1 = nn.Conv2d(3, round(32 * k), kernel_size=3, stride=1, padding=0,
                               bias=not use_bn)

        scale = calculate_scale(self.conv1.weight.data)
        torch.nn.init.uniform_(self.conv1.weight.data, -scale, scale)
        if self.conv1.bias is not None:
            self.conv1.bias.data.zero_()
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(round(32 * k))
        if use_relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = nn.PReLU(round(32 * k))

        block = block if block is not None else BasicBlock
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, round(64 * k), layers[0])
        self.layer2 = self._make_layer(block, round(128 * k), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 * k), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 * k), layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(round(12800 * k), 512, bias = True)

        scale = calculate_scale(self.fc.weight.data)
        torch.nn.init.uniform_(self.fc.weight.data, -scale, scale)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()

        if pretrained:
            # Put your path
            weights = torch.load('/path/to/pretrained/weights.pth', map_location='cpu')
            self.load_state_dict(weights)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding=0,
                                bias=not self.use_bn))

        scale = calculate_scale(layers[-1].weight.data)
        torch.nn.init.uniform_(layers[-1].weight.data, -scale, scale)
        if layers[-1].bias is not None:
            layers[-1].bias.data.zero_()
        if self.use_bn:
            layers.append(nn.BatchNorm2d(planes))
        if use_relu:
            layers.append(nn.ReLU(inplace = True))
        else:
            layers.append(nn.PReLU(planes))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.inplanes = planes
        for i in range(0, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def test():
    model = ResNetCaffe( [1, 2, 5, 3], BasicBlock, pretrained=True)
    test_tensor = torch.rand(2, 3, 112, 112)
    print(model(test_tensor).size())
        