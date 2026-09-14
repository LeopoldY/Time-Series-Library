"""Keep the forecast backbone and append a small learnable hidden-layer head."""
import importlib
from types import SimpleNamespace
import torch
from torch import nn


class SafeBinaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model not in ('Informer', 'iTransformer', 'PatchTST'):
            raise ValueError('Unsupported SAFE backbone')
        config = SimpleNamespace(**vars(args))
        # Retain forecast branches including Informer decoder and distillation.
        config.task_name = 'long_term_forecast'
        config.enc_in = config.dec_in = config.c_out = 4
        config.pred_len = config.label_len = 1
        config.embed, config.freq, config.activation = 'timeF', 'h', 'gelu'
        config.distil = True
        self.backbone = importlib.import_module('models.' + args.model).Model(config)
        self.head = nn.Sequential(nn.Linear(4, args.head_hidden), nn.GELU(),
                                  nn.Dropout(args.head_dropout), nn.Linear(args.head_hidden, 1))

    def forward(self, history, history_mark, decoder_mark):
        # The future counts are never passed into this module.
        decoder = torch.cat((history[:, -1:, :], torch.zeros_like(history[:, -1:, :])), dim=1)
        forecast = self.backbone(history, history_mark, decoder, decoder_mark)
        if forecast.ndim != 3 or forecast.shape[1:] != (1, 4):
            raise RuntimeError(f'Expected forecast [B,1,4], got {forecast.shape}')
        return self.head(forecast[:, -1, :])  # [B,1] logits, no sigmoid before BCE
