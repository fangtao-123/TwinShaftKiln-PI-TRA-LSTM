# -*- coding: utf-8 -*-
from ..common import *
from ..config import CONFIG
from ..model_blocks import ResidualFFN, AdditiveAttention, GeneralAttention, _BaseRNN

# -------------------------
# 3. 模型族
# -------------------------
class LSTM_Base(nn.Module):
    def __init__(self, input_dim=7, hidden=64, layers=1, dropout=0.0, stronger=False):
        super().__init__()
        self.enc = _BaseRNN(input_dim, hidden, layers, dropout, False, stronger)
        self.fc = nn.Linear(self.enc.out_dim, 1)

    def forward(self, x):
        H = self.enc.encode(x)
        return self.fc(H[:, -1, :])


class BiLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden=64, layers=1, dropout=0.0, stronger=False):
        super().__init__()
        self.enc = _BaseRNN(input_dim, hidden, layers, dropout, True, stronger)
        self.fc = nn.Linear(self.enc.out_dim, 1)

    def forward(self, x):
        H = self.enc.encode(x)
        return self.fc(H[:, -1, :])


class ReAttentionHead(nn.Module):
    """
    Re-attention head for attention consistency loss.
    Query: predicted output y_hat
    Key/Value: encoder hidden states H

    Implements general attention:
        alpha_re = softmax(q' W_r K^T)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # y_hat: (B, 1) -> project to hidden_dim
        self.q_proj = nn.Linear(1, hidden_dim, bias=False)
        # general attention weight
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, y_hat, H):
        """
        y_hat: (B, 1)
        H:     (B, L, hidden_dim)
        return:
            alpha_re: (B, L)
        """
        # q': (B, hidden_dim)
        q = self.q_proj(y_hat)  # (B, hidden)
        qW = self.W(q).unsqueeze(1)  # (B, 1, hidden)

        # score: (B, L)
        score = torch.bmm(qW, H.transpose(1, 2)).squeeze(1)
        alpha_re = torch.softmax(score, dim=1)

        return alpha_re


class TRA_LSTM(nn.Module):
    """
    Encoder–decoder + temporal attention (architecture-aligned).
    Decoder input: concat(ctx, x_last)
    Decoder init: encoder (h_n, c_n)

    Plus (for Loss Layer): optional re-attention weights alpha_re computed from y_hat.
    """

    def __init__(self, input_dim=7, hidden=64, layers=1, dropout=0.0,
                 stronger=False, attn_drop=0.0, use_reattn=True,
                 beta_att: float = 0.0):  # <-- 新增：兼容 build_model 传参（但 TRA 不用它）
        super().__init__()
        self.enc = _BaseRNN(input_dim, hidden, layers, dropout, False, stronger)

        self.proj_q = nn.Linear(input_dim, hidden)
        self.attn = AdditiveAttention(hidden, hidden, hidden, attn_drop)

        self.dec_lstm = nn.LSTM(
            input_size=hidden + input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, 1)

        self.last_alpha = None
        self.last_alpha_re = None

        self.use_reattn = use_reattn
        self.re_head = ReAttentionHead(hidden) if use_reattn else None

    def forward(self, x, savg=None, return_attn=False):
        # savg is ignored in TRA-LSTM (kept for unified pipeline compatibility)

        # Encoder
        H, (h_n, c_n) = self.enc.encode(x, return_states=True)

        # Temporal attention (query from x_last)
        x_last = x[:, -1, :]
        q = self.proj_q(x_last)
        ctx, alpha = self.attn(H, q)
        self.last_alpha = alpha.detach()

        # Decoder (1-step)
        x_dec = torch.cat([ctx, x_last], dim=-1).unsqueeze(1)
        dec_out, _ = self.dec_lstm(x_dec, (h_n, c_n))
        y_hat = self.fc(dec_out[:, -1, :])  # (B,1)

        # Re-attention weights (optional)
        if self.use_reattn:
            alpha_re = self.re_head(y_hat, H)  # (B,L)
            self.last_alpha_re = alpha_re.detach()
        else:
            alpha_re = None
            self.last_alpha_re = None

        if return_attn:
            return y_hat, alpha, alpha_re
        return y_hat


class PI_LSTM(LSTM_Base):
    pass


class PI_TRA_LSTM(TRA_LSTM):
    """
    新增：PI-TRA-LSTM
    结构 = TRA-LSTM（encoder-decoder） + 物理损失（在训练管线里通过 PHYS_CLASSES 自动启用）
    """
    pass


class GRU_Base(nn.Module):
    def __init__(self, input_dim=7, hidden=64, layers=1, dropout=0.0, bidirectional=False, stronger=True):
        super().__init__()
        self.in_ln = nn.LayerNorm(input_dim)
        self.rnn = nn.GRU(
            input_dim, hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.ffn = ResidualFFN(out_dim, drop=dropout) if stronger else nn.Identity()
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.in_ln(x)
        h, _ = self.rnn(x)
        h = self.ffn(h)
        return self.fc(h[:, -1, :])


class CNNLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden=64, drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=2, dilation=2)
        self.ln = nn.LayerNorm(hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = torch.relu(self.conv(z))[:, :, :x.size(1)].transpose(1, 2)
        z = self.ln(z)
        z, _ = self.lstm(z)
        return self.fc(z[:, -1, :])


class TPALSTM(nn.Module):
    def __init__(self, input_dim=7, hidden=64, drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.proj = nn.Linear(input_dim, hidden)
        self.attn = GeneralAttention(hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        H, _ = self.lstm(x)
        q = self.proj(x[:, -1, :])
        ctx, _ = self.attn(q, H, H)
        return self.fc(ctx)


# 物理损失启用的模型集合
PHYS_CLASSES = (PI_LSTM, PI_TRA_LSTM)
