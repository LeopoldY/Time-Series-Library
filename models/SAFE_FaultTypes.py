"""Regression pretraining followed by frozen-backbone multi-label prediction."""
import importlib
from types import SimpleNamespace
import torch
from torch import nn


class FaultTypeModel(nn.Module):
    def __init__(self, model, seq_len, channels, types, d_model=64):
        super().__init__()
        if model not in ('Informer', 'iTransformer', 'PatchTST'):
            raise ValueError('Unsupported backbone')
        config = SimpleNamespace(task_name='long_term_forecast', seq_len=seq_len,
            pred_len=1, label_len=1, enc_in=channels, dec_in=channels, c_out=channels,
            d_model=d_model, n_heads=4, d_ff=d_model*4, e_layers=2, d_layers=1,
            factor=3, dropout=.1, embed='timeF', freq='h', activation='gelu',
            distil=True, patch_len=max(1, seq_len//2), stride=max(1, seq_len//4))
        self.backbone = importlib.import_module('models.'+model).Model(config)
        self.head = nn.Sequential(nn.Linear(channels*3, d_model), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(d_model, types+1))

        self.backbone_frozen = False

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.zero_grad(set_to_none=True)
        self.backbone_frozen = True
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forecast(self, x, xm, dm):
        decoder = torch.cat([x[:, -1:], torch.zeros_like(x[:, -1:])], dim=1)
        return self.backbone(x, xm, decoder, dm)[:, -1]

    def forward(self, x, xm, dm):
        if self.backbone_frozen:
            with torch.no_grad():
                forecast = self.forecast(x, xm, dm)
        else:
            forecast = self.forecast(x, xm, dm)
        return self.head(torch.cat([forecast, x[:, -1], x.mean(1)], dim=1))
