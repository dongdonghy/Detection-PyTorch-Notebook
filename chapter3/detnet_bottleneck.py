from torch import nn
class DetBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False), 
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, 
                               dilation=2, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, 1, bias=False),
                nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        if self.extra:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out

