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
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GroupGRU(input_size=input_size, hidden_size=hidden_size, groups=groups, bidirectional=False)

    def forward(self, x, hidden_state):
        """Intra RNN"""
        # [B, C//2, T, F]
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
        inter_x, hidden_state = self.inter_rnn(inter_x, hidden_state)  # [B * F, T, C]
        inter_out = torch.reshape(inter_x, shape=(B, F, T, C))
        inter_out = torch.add(inter_out, x)

        dpe_out = inter_out.permute(0, 3, 2, 1)  # [B, C, F, T]

        return dpe_out, hidden_state


class GroupGRU(nn.Module):
    def __init__(self, input_size, hidden_size, groups, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.groups = groups
        self.gru_list = []
        self.fc_list = []

        for _ in range(groups):
            self.gru_list.append(getattr(nn, "GRU")(input_size=input_size // groups, hidden_size=hidden_size,
                                                    num_layers=num_layers, batch_first=batch_first,
                                                    bidirectional=bidirectional))
            self.fc_list.append(getattr(nn, "Linear")(hidden_size, 2))

    def forward(self, x, hidden_state):
        # x: [B * F, T, C]
        gru_outputs = []
        out_states = []
        final_outputs = []

        B, T, _ = x.shape
        x = torch.reshape(x, shape=(B, T, self.groups, -1))
        for idx, gru in enumerate(self.gru_list):

            y, h = gru(x[:, :, idx, :], hidden_state[idx])
            gru_outputs.append(y)
            out_states.append(h)

        for idx, fc in enumerate(self.fc_list):
            y = fc(gru_outputs[idx])
            final_outputs.append(y)

        final_outputs = torch.cat(final_outputs, dim=2)
        return final_outputs, out_states
