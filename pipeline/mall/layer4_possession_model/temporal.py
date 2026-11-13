"""Experimental offline temporal convolutions for sharp possession boundaries.

Same inputs/outputs as the GRU. Symmetric dilated convolutions cover both sides
of a contact; residual paths preserve local detail. Layer normalization acts on
channels within each frame, not over the length of an inference chunk.
"""


def build_net(dim, hidden=128, blocks=8):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            h = self.conv1(self.act(self.norm1(x)).transpose(1, 2)).transpose(1, 2)
            h = self.drop(h)
            h = self.conv2(self.act(self.norm2(h)).transpose(1, 2)).transpose(1, 2)
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
