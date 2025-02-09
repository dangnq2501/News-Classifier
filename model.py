import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel

class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.5, max_length=768, num_classes=4):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('./local_directory/')
        for param in self.bert.parameters():
           param.require_grad = False
        self.fcn = nn.Sequential(
           nn.Linear(max_length, hidden_size),
           nn.Dropout(dropout),
           nn.Linear(hidden_size, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  
        logits = self.fcn(self.dropout(pooled_output))
        return logits
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

class GRUnode(nn.Module):
  def __init__(self, input_size, hidden_size, bias=True):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.x2h = nn.Linear(input_size, 3*hidden_size, bias=self.bias)
    self.h2h = nn.Linear(hidden_size, 3*hidden_size, bias=self.bias)
    self.reset_parameters()

  def reset_parameters(self):
    std = 1.0 / np.sqrt(self.hidden_size)
    for w in self.parameters():
      w.data.uniform_(-std, std)

  def forward(self, X, hx=None):
    if hx is None:
      hx = Variable(input.new_zeros(X.size(0), self.hidden_size))

    x_t = self.x2h(X)
    h_t = self.h2h(hx)

    x_renet, x_upd, x_new = x_t.chunk(3, 1)
    h_renet, h_upd, h_new = h_t.chunk(3, 1)

    renet_gate = torch.sigmoid(x_renet+h_renet)
    update_gate = torch.sigmoid(x_upd +h_upd)
    new_gate = torch.tanh(x_new + (renet_gate * h_new))
    return update_gate * hx + (1-update_gate) * new_gate

class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers, bias, device="cpu"):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.bias = bias
    self.device = device

    self.gru_layers = nn.ModuleList()
    self.gru_layers.append(GRUnode(self.input_size, self.hidden_size, self.bias))
    for _ in range(self.num_layers-1):
      self.gru_layers.append(GRUnode(self.hidden_size, self.hidden_size, self.bias))
    self.linear = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x, hx=None):
    if hx is None:
      h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
    else:
      h0 = hx

    outs = []
    hidden = list()
    for layer in range(self.num_layers):
      hidden.append(h0[layer, :, :])

    for t in range(x.size(1)):
      for i in range(self.num_layers):
        if i:
          hidden_layer = self.gru_layers[i](hidden[i-1], hidden[i])
        else:
          hidden_layer = self.gru_layers[i](x[:, t, :], hidden[i])
        hidden[i] = hidden_layer
      outs.append(hidden[-1])

    outs = outs[-1].squeeze()
    return self.linear(outs)
