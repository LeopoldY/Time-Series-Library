"""Keep the forecast backbone and append a small learnable hidden-layer head."""
import importlib
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from torch import nn
from models.SAFE_Imbalance import HistoryAwareHead


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
        self.head_type = getattr(args, "head_type", "mlp")
        if self.head_type == "history":
            self.head = HistoryAwareHead(args.head_hidden, args.head_dropout)
        elif self.head_type != "mlp":
            raise ValueError("Unknown head type")
        self.backbone_frozen = False

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.zero_grad(set_to_none=True)
        self.backbone_frozen = True
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            # Freeze BatchNorm buffers and disable backbone dropout too.
            self.backbone.eval()
        return self

    def forecast(self, history, history_mark, decoder_mark):
        # The future counts are never passed into this module.
        decoder = torch.cat((history[:, -1:, :], torch.zeros_like(history[:, -1:, :])), dim=1)
        forecast = self.backbone(history, history_mark, decoder, decoder_mark)
        if forecast.ndim != 3 or forecast.shape[1:] != (1, 4):
            raise RuntimeError(f'Expected forecast [B,1,4], got {forecast.shape}')
        return forecast[:, -1, :]  # [B,4] raw next-hour counts

    def forward(self, history, history_mark, decoder_mark):
        with torch.no_grad() if self.backbone_frozen else nullcontext():
            forecast = self.forecast(history, history_mark, decoder_mark)
        if self.head_type == "history":
            return self.head(forecast, history)
        return self.head(forecast)  # [B,1] logits
