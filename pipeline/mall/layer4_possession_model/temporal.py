"""Experimental offline temporal convolutions for sharp possession boundaries.

Same inputs/outputs as the GRU. Symmetric dilated convolutions cover both sides
of a contact; residual paths preserve local detail. Layer normalization acts on
channels within each frame, not over the length of an inference chunk.
"""


def build_net(dim, hidden=128, blocks=8):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DilatedConv(nn.Module):
        """Kernel-3 dilated convolution with zero padding, on (batch, time, channels).

        Written as one linear map of [x(t - d), x(t), x(t + d)]: the same function as
        nn.Conv1d(padding=d, dilation=d), but plain matrix products, so deterministic
        training does not fall back to cuDNN's ~20x slower deterministic dilated kernels.
        """

        def __init__(self, dilation):
            super().__init__()
            self.d = dilation
            self.lin = nn.Linear(3 * hidden, hidden)

        def forward(self, x):
            d, n = self.d, x.shape[1]
            p = F.pad(x, (0, 0, d, d))
            return self.lin(torch.cat([p[:, :n], p[:, d:d + n], p[:, 2 * d:2 * d + n]], -1))

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.conv1 = DilatedConv(dilation)
            self.conv2 = DilatedConv(dilation)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            h = self.drop(self.conv1(self.act(self.norm1(x))))
            h = self.conv2(self.act(self.norm2(h)))
            return x + self.drop(h)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(dim, hidden)
            self.blocks = nn.Sequential(*(Block(2 ** i) for i in range(blocks)))
            self.norm = nn.LayerNorm(hidden)
            self.poss = nn.Linear(hidden, 1)
            self.change = nn.Linear(hidden, 3)

        def forward(self, x):
            h = self.norm(self.blocks(self.inp(x)))
            return self.poss(h).squeeze(-1), self.change(h)

    return Net()
