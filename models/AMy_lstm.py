import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard LSTM baseline for multivariate forecasting (TSLib-style).

    Uses encoder input (x_enc) + time features (x_mark_enc), encodes with an LSTM,
    and projects the final hidden state to the forecast horizon.

    Inputs:
        x_enc:      [B, seq_len, enc_in]
        x_mark_enc: [B, seq_len, d_mark]
    Output:
        y_hat:      [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # Hyperparameters
        self.hidden_size = int(getattr(configs, "lstm_hidden", 128))
        self.num_layers = int(getattr(configs, "lstm_layers", 1))
        self.dropout = float(getattr(configs, "dropout", 0.0))

        # Time feature dim: try to read from configs, otherwise infer at runtime
        self.d_mark = int(getattr(configs, "d_mark", 0))

        # LSTM input size = variables + time features
        input_size = self.enc_in + self.d_mark

        # LSTM dropout only works if num_layers > 1
        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Optional dropout for single-layer case (applies to representation)
        self.out_dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        self.projection = nn.Linear(self.hidden_size, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        # Infer d_mark if not provided
        if self.d_mark == 0:
            self.d_mark = x_mark_enc.size(-1)
            # NOTE: if this happens, input_size in LSTM was built wrong.
            # So we enforce configs.d_mark to be set for correctness.
            raise ValueError(
                "configs.d_mark was 0/missing. Set configs.d_mark = x_mark_enc.size(-1) when building the model."
            )

        x = torch.cat([x_enc, x_mark_enc], dim=-1)  # [B, T, enc_in + d_mark]

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]               # [B, hidden_size]
        h_last = self.out_dropout(h_last)

        out = self.projection(h_last)  # [B, pred_len*c_out]
        out = out.view(x_enc.size(0), self.pred_len, self.c_out)

        return out
