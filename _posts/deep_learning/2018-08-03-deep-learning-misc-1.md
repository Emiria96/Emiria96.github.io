---
layout: post
title: 用Pytorch训练ResNet的一点小杂记
author: Scalsol
category: [ dl_misc ]
---

因为一直记不住模型构建的步骤，所以今天又特意重写了一遍ResNet。下面记一下一点细节。

### 模型细节

首先介绍两个基础模块：**BasicBlock**和**BottleNeck**。其中**BasicBlock**的细节如下：

{% highlight py %}
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
{% endhighlight%}
也就是残差块的结构是**3x3卷积+bn+relu+3x3卷积+bn**，然后进行残差连接，最后再进行一次**relu**。下面是一张示意图：
![BasicBlock](/assets/images/basicblock.png)

而BottleNeck，正如其名中间的一层比较细。结构如下：
{% highlight py %}
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
{% endhighlight%}
![BottleNeck](/assets/images/bottleneck.png)

还有要注意一点的是不同ResBlock间的维度变化。这时若要将一个256x56x56的特征图转变成一个512x28x28的特征图，需要减小特征图的大小和增加维度。这时候的downsample需要特殊处理一下，这里采用的方法是使用stride为2的一维卷积核+bn的方法作为downsample。论文里还讨论了别的方法，这里就不讨论了。还需要注意的是第一个ResBlock的特征图长宽是没有变换的。整体的网络结构代码如下所示：
{% highlight py %}
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
{% endhighlight%}

至于CIFAR上的网络结构，则是将最开始的7x7conv(stride=2, pad=3)+bn+relu+maxpool改为了3x3conv+bn+relu。

这样就差不多把ResNet的结构给讲的差不多了，接下来准备处理一下DenseNet还有`net.parameters`、`net.named_parameters`、`net.modules()`、`net.named_modules()`、`net.params()`等函数。因为这些写代码看代码的时候还经常会碰到，所以准备记下笔记免得之后忘掉。