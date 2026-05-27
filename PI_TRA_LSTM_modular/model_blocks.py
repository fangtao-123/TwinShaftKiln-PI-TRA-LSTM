# -*- coding: utf-8 -*-
from .common import *

# -------------------------
# 2. 模块与注意力
# -------------------------
class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden_mul=2, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_mul)
        self.act = nn.GELU()
        self.do = nn.Dropout(drop)
        self.fc2 = nn.Linear(dim * hidden_mul, dim)

    def forward(self, x):
        z = self.ln(x)
        z = self.fc1(z)
        z = self.act(z)
        z = self.do(z)
        z = self.fc2(z)
        z = self.do(z)
        return x + z


class AdditiveAttention(nn.Module):
    def __init__(self, h_dim, q_dim, attn_dim, attn_drop=0.0):
        super().__init__()
        self.W_h = nn.Linear(h_dim, attn_dim, bias=False)
        self.W_q = nn.Linear(q_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, H, q):
        H_ = self.W_h(H)
        q_ = self.W_q(q).unsqueeze(1)
        e = torch.tanh(H_ + q_)
        score = self.v(e).squeeze(-1)
        alpha = torch.softmax(score, dim=1)
        alpha = self.drop(alpha)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        ctx = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return ctx, alpha


class GeneralAttention(nn.Module):
    def __init__(self, q_dim, k_dim):
        super().__init__()
        self.W = nn.Linear(q_dim, k_dim, bias=False)

    def forward(self, q, K, V):
        qW = self.W(q).unsqueeze(1)
        score = torch.bmm(qW, K.transpose(1, 2)).squeeze(1)
        alpha = torch.softmax(score, dim=1)
        ctx = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        return ctx, alpha


class _BaseRNN(nn.Module):
    """
    FIX: 支持 return_states=True，供 TRA_LSTM decoder 使用
    """

    def __init__(self, input_dim, hidden, layers, dropout=0.0, bidirectional=False, stronger=False):
        super().__init__()
        self.in_ln = nn.LayerNorm(input_dim)
        self.rnn = nn.LSTM(
            input_dim, hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.post = ResidualFFN(out_dim, drop=dropout) if stronger else nn.Identity()
        self.out_dim = out_dim

    def encode(self, x, return_states: bool = False):
        x = self.in_ln(x)
        H, (h_n, c_n) = self.rnn(x)
        H = self.post(H)
        if not return_states:
            return H
        return H, (h_n, c_n)


