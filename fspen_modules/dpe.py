import torch
import torch.nn as nn


class DPE(nn.Module):
    def __init__(self, input_size, width, hidden_size, groups, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(32, 16)
        # self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = PathExtensionGRU(input_size=input_size, hidden_size=hidden_size, groups=groups,
                                          bidirectional=False)

    def forward(self, x, hidden_states):
        """x: (B,C,T,F)"""
        """Intra RNN"""
        x = x.permute(0, 2, 3, 1).contiguous()
        B, T, F, C = x.shape
        intra_x = torch.reshape(x, shape=(B * T, F, C))
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = torch.reshape(intra_x, shape=(B, T, F, C))
        # intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        """Inter RNN"""
        x = intra_out.permute(0, 3, 1, 2)  # (B,C,T,F)
        inter_x, hidden_states = self.inter_rnn(x, hidden_states)  # (B,C,T,F)
        dpe_out = torch.add(inter_x, x)  # (B,C,T,F)

        return dpe_out, hidden_states


class PathExtensionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, groups, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.groups = groups
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnns = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(groups):
            self.rnns.append(
                nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional))
            self.fcs.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x, hidden_states):
        """x: (B,C,T,F)"""
        B, C, T, F = x.shape
        x_chunks = torch.chunk(x, self.groups, dim=3)  # 8 x (B,C,T,F//8)
        y_chunks = []
        h_chunks = []

        for i in range(self.groups):
            x_chunk = x_chunks[i].permute(0, 3, 2, 1).reshape(-1, T, C).contiguous()
            h_chunk = hidden_states[i]
            y_chunk, h_chunk = self.rnns[i](x_chunk, h_chunk)
            y_chunk = self.fcs[i](y_chunk)
            y_chunk = y_chunk.reshape(B, -1, T, C).permute(0, 3, 2, 1).contiguous()
            y_chunks.append(y_chunk)
            h_chunks.append(h_chunk)
        y = torch.cat(y_chunks, dim=3)
        return y, h_chunks
