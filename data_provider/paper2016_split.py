"""Deterministic blocked ten-fold CV; every sample stays within one role."""
import numpy as np
import pandas as pd


def row_roles(n, fold):
    if not 0 <= fold < 10:
        raise ValueError('fold must be in [0, 9]')
    blocks = np.minimum(np.arange(n) * 10 // max(1, n), 9)
    return np.where(blocks == fold, 'test', np.where(blocks == (fold + 1) % 10, 'val', 'train'))


def split_targets(dates, seq_len, split, fold):
    if seq_len < 1 or split not in ('train', 'val', 'test'):
        raise ValueError('Invalid sequence length or split')
    dates = pd.DatetimeIndex(dates)
    roles = row_roles(len(dates), fold)
    candidates = np.arange(seq_len, len(dates), dtype=np.int64)
    forbidden = (roles != split).astype(int)
    prefix = np.r_[0, forbidden.cumsum()]
    inside = prefix[candidates + 1] == prefix[candidates - seq_len]
    gaps = np.r_[0, np.asarray(dates[1:]-dates[:-1] != pd.Timedelta(hours=1), dtype=int)].cumsum()
    continuous = gaps[candidates] == gaps[candidates-seq_len]
    return candidates[inside & continuous]
