from torch import nn

class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim),
          nn.Sigmoid(),
          nn.Linear(hid_dim, out_dim),
          nn.Sigmoid()
)
    def forward(self, x):
        y = self.layer(x)
        return y
