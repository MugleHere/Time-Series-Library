import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        last = x_enc[:, -1:, :]                 # [B, 1, C]
        out = last.repeat(1, self.pred_len, 1)  # [B, pred_len, C]
        return out
