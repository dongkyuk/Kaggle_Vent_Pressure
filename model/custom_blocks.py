import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self, input_size, output_size, act=None, norm=None, skip_connect=False):
        super(MlpBlock, self).__init__()
        self.skip_connect = skip_connect
        self.fc = nn.Linear(input_size, output_size)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'swish':
            self.act = nn.SiLU()
        elif act is None:
            self.act = nn.Identity()

        if norm == "bn":
            self.norm = nn.BatchNorm1d(output_size)
        elif norm == "ln":
            self.norm = nn.LayerNorm(output_size)
        elif norm is None:
            self.norm = nn.Identity()

    def forward(self, x):
        out = self.fc(x)
        if self.skip_connect:
            out += x
        out = self.norm(out)
        out = self.act(out)
        return out


class MlpEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, act, norm):
        super(MlpEncoder, self).__init__()
        head = MlpBlock(input_size, hidden_size, act, norm)
        hidden = MlpBlock(hidden_size, hidden_size, act, norm, skip_connect=True)
        tail = MlpBlock(hidden_size, output_size, act, norm)
        self.encoder = [head] + [hidden for _ in range(num_hidden_layers)] + [tail]
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)
