import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class LSTMnode(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xh = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hx):
        h_t, c_t = hx
        gates = self.xh(x) + self.hh(h_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        i_t, f_t, c_t, o_t = torch.sigmoid(input_gate), torch.sigmoid(forget_gate), torch.tanh(cell_gate), torch.sigmoid(output_gate)
        cy = c_t * f_t + i_t * c_t
        hy = torch.tanh(cy) * o_t
        return hy, cy


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_net = nn.ModuleList([LSTMnode(input_size if i == 0 else hidden_size, hidden_size, bias) for i in range(num_layers)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx=None):
        if hx is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            hx = [(h0[i], c0[i]) for i in range(self.num_layers)]

        outs = []
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                hx[layer] = self.lstm_net[layer](x[:, t, :] if layer == 0 else hx[layer - 1][0], hx[layer])
            outs.append(hx[-1][0])
        outs = torch.stack(outs, dim=1)
        return self.linear(outs[:, -1, :])