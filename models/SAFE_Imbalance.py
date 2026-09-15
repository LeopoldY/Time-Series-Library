"""History-aware binary head and ASL (Ridnik et al., ICCV 2021) adaptation.

The history summary is a project-specific design, not an architecture from ASL.
All outputs are one binary logit for any-level occurrence in the next hour.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class HistoryAwareHead(nn.Module):
    def __init__(self, hidden=16, dropout=.1):
        super().__init__()
        self.forecast_branch = nn.Sequential(nn.Linear(4, hidden), nn.GELU())
        self.history_branch = nn.Sequential(nn.Linear(20, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(hidden, 1)
        self.shortcut = nn.Linear(24, 1, bias=False)

    @staticmethod
    def history_features(history):
        # Last observed count, mean, max, occurrence rate, normalized time since
        # last occurrence; absence throughout the lookback is encoded as 1.
        length = history.shape[1]
        occurred = history > 0
        steps = torch.arange(1, length+1, device=history.device).view(1, length, 1)
        last_position = (occurred * steps).amax(dim=1)
        recency = (length-last_position).to(history.dtype)/length
        return torch.cat((history[:, -1].log1p(), history.mean(1).log1p(),
                          history.amax(1).log1p(), occurred.to(history.dtype).mean(1), recency), dim=1)

    def initialize_prior(self, positive, negative):
        if positive <= 0 or negative <= 0:
            raise ValueError('Both training classes are required')
        with torch.no_grad():
            self.output.bias.fill_(math.log(positive/negative))

    def forward(self, forecast, history):
        # Signed log1p preserves negative regression predictions rather than clipping.
        forecast = forecast.sign() * forecast.abs().log1p()
        context = self.history_features(history)
        fused = self.fusion(torch.cat((self.forecast_branch(forecast),
                                       self.history_branch(context)), dim=1))
        return self.output(fused) + self.shortcut(torch.cat((forecast, context), dim=1))


class AsymmetricBinaryLoss(nn.Module):
    """Binary sigmoid ASL, mean reduction, full gradients through focal weights.

    Stable log-sigmoid avoids losing positive gradients at large negative logits.
    No pos_weight or over-sampling is combined with this loss.
    """
    def __init__(self, gamma_neg=4., gamma_pos=0., clip=.05):
        super().__init__()
        if not all(math.isfinite(v) for v in (gamma_neg, gamma_pos, clip)) or min(gamma_neg, gamma_pos) < 0 or not 0 <= clip < 1:
            raise ValueError('Invalid ASL parameters')
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits, targets):
        p = logits.sigmoid()
        log_pos = F.logsigmoid(logits)
        log_neg = F.logsigmoid(-logits)
        if self.clip:
            log_neg = torch.logaddexp(log_neg, logits.new_tensor(math.log(self.clip))).clamp(max=0.)
        p_minus = (-log_neg.expm1()).clamp(min=0., max=1.)
        positive = -(1-p).pow(self.gamma_pos)*log_pos
        negative = -p_minus.pow(self.gamma_neg)*log_neg
        return torch.where(targets.bool(), positive, negative).mean()


def select_threshold(y, scores, policy='fixed', fixed=.5, beta=1.):
    """Exact validation F-beta sweep; ties prefer the higher threshold (fewer FPs)."""
    y, scores = np.asarray(y).reshape(-1), np.asarray(scores).reshape(-1)
    if (len(y) != len(scores) or not len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not math.isfinite(beta) or beta <= 0 or not 0 < fixed < 1):
        raise ValueError('Invalid threshold inputs')
    if policy not in ('fixed', 'val_fbeta'):
        raise ValueError('Unknown threshold policy')
    if policy == 'fixed' or len(np.unique(y)) < 2:
        return {'threshold': float(fixed), 'policy': policy, 'source': 'fixed',
                'reason': 'configured fixed threshold' if policy == 'fixed' else 'single-class validation; no tuning', 'beta': beta}
    order = np.argsort(-scores, kind='stable')
    s, truth = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.cumsum(truth)[ends]
    fp = ends+1-tp
    fn = truth.sum()-tp
    fbeta = (1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)
    best = int(np.argmax(fbeta))
    return {'threshold': float(s[ends[best]]), 'policy': policy, 'source': 'validation',
            'beta': beta, 'validation_fbeta': float(fbeta[best]), 'validation_windows': len(y),
            'tie_break': 'highest threshold', 'candidates': len(ends)}
