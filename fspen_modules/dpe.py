import torch
import torch.nn as nn


class DPE(nn.Module):
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(32, 16)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GroupGRU(input_size=input_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, x):
        """Intra RNN"""
        x = x.permute(0, 2, 3, 1).contiguous()
        B, T, F, C = x.shape
        intra_x = torch.reshape(x, shape=(B * T, F, C))
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = torch.reshape(intra_x, shape=(B, T, F, C))
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        """Inter RNN"""
        x = intra_out.permute(0, 2, 1, 3)  # [B, F_c, T, C]
        inter_x = torch.reshape(x, shape=(B * F, T, C))
        inter_x = self.inter_rnn(inter_x)[0]  # [B * F, T, C]
        inter_out = torch.reshape(inter_x, shape=(B, F, T, C))
        inter_out = torch.add(inter_out, x)

        dpe_out = inter_out.permute(0, 3, 2, 1)  # [B, C, F, T]

        return dpe_out

class GroupGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn3 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn4 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn5 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn6 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn7 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn8 = nn.GRU(input_size // 8, hidden_size // 8, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 2)
        self.fc6 = nn.Linear(2, 2)
        self.fc7 = nn.Linear(2, 2)
        self.fc8 = nn.Linear(2, 2)

    def forward(self, x, h=None):
        h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)

        x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x, chunks=8, dim=-1)
        h1, h2, h3, h4, h5, h6, h7, h8 = torch.chunk(h, chunks=8, dim=-1)

        h1, h2, h3, h4, h5, h6, h7, h8 = h1.contiguous(), h2.contiguous(), h3.contiguous(), h4.contiguous(), h5.contiguous(), h6.contiguous(), h7.contiguous(), h8.contiguous()

        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y3, h3 = self.rnn3(x3, h3)
        y4, h4 = self.rnn4(x4, h4)
        y5, h5 = self.rnn5(x5, h5)
        y6, h6 = self.rnn6(x6, h6)
        y7, h7 = self.rnn7(x7, h7)
        y8, h8 = self.rnn8(x8, h8)

        y1 = self.fc1(y1)
        y2 = self.fc2(y2)
        y3 = self.fc3(y3)
        y4 = self.fc4(y4)
        y5 = self.fc5(y5)
        y6 = self.fc6(y6)
        y7 = self.fc7(y7)
        y8 = self.fc8(y8)

        y = torch.concat((y1, y2, y3, y4, y5, y6, y7, y8), dim=-1)
        h = torch.concat((h1, h2, h3, h4, h5, h6, h7, h8), dim=-1)
        return y, h
