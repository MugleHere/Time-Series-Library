import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Multivariate linear regression baseline for forecasting.

    Predicts future values as a linear function of the past window:
        vec(Y_future) = W * vec(X_past) + b

    - Works for features="M" (predict all channels).
    - Supports pred_len >= 1.
    - Learns cross-channel dependencies because all channels are in the input vector.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)

        in_dim = self.seq_len * self.enc_in
        out_dim = self.pred_len * self.c_out

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]
        B, T, C = x_enc.shape
        # Defensive: ensure expected lengths (useful when debugging data_provider)
        # (Remove later if you prefer)
        # assert T == self.seq_len and C == self.enc_in, (T, C, self.seq_len, self.enc_in)

        x = x_enc.reshape(B, T * C)          # [B, seq_len*enc_in]
        y = self.linear(x)                  # [B, pred_len*c_out]
        y = y.reshape(B, self.pred_len, self.c_out)
        return y
