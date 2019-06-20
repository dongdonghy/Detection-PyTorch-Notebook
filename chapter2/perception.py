import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = x.matmul(self.w)
        y = x + self.b.expand_as(x)
        return y

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        y = torch.sigmoid(x)
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y
