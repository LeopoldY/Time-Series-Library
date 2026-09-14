"""Hourly next-step binary labels; windows never cross a cleaned-data gap."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features

FEATURES = ['w_level1', 'w_level2', 'w_level3', 'w_level4']


class SafeBinaryDataset(Dataset):
    def __init__(self, path, seq_len, split):
        if split not in ('train', 'val', 'test') or seq_len < 1:
            raise ValueError('Invalid split or seq_len')
        df = pd.read_csv(Path(path))
        dates = pd.DatetimeIndex(pd.to_datetime(df['date'], errors='raise'))
        if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('Dates must be valid, strictly increasing and unique')
        values = df[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Alarm counts must be finite and nonnegative')
        self.values = values
        self.labels = (values > 0).any(axis=1).astype(np.float32)
        self.marks = time_features(dates, freq='h').transpose(1, 0).astype(np.float32)
        self.dates, self.seq_len = dates, seq_len
        n = len(df)
        train_end, val_end = int(n * .7), n - int(n * .2)
        lo, hi = {'train': (0, train_end), 'val': (train_end, val_end),
                  'test': (val_end, n)}[split]
        # gap[k] marks the edge between row k-1 and k. Include the target edge.
        gap = np.zeros(n, dtype=np.int64)
        gap[1:] = np.asarray((dates[1:] - dates[:-1]) != pd.Timedelta(hours=1))
        cumulative = gap.cumsum()
        candidates = np.arange(max(lo, seq_len), hi, dtype=np.int64)
        self.targets = candidates[cumulative[candidates] == cumulative[candidates-seq_len]]
        self.summary = {'split': split, 'rows': n, 'target_start': lo, 'target_end': hi,
                        'windows': len(self.targets),
                        'gap_rejected': len(candidates)-len(self.targets),
                        'positive': int(self.labels[self.targets].sum())}
        self.summary['negative'] = len(self.targets) - self.summary['positive']
        if not len(self.targets):
            raise ValueError(f'No continuous windows for {split}, seq_len={seq_len}')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        t = int(self.targets[index])
        return (torch.from_numpy(self.values[t-self.seq_len:t].copy()),
                torch.from_numpy(self.marks[t-self.seq_len:t].copy()),
                torch.from_numpy(self.marks[t-1:t+1].copy()),
                torch.tensor([self.labels[t]], dtype=torch.float32), t)
