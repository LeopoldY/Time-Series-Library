"""Pool per-device windows into one cluster dataset without crossing device boundaries."""
import numpy as np
import pandas as pd
from data_provider.safe_binary_data import FEATURES, SafeBinaryDataset
from data_provider.paper2016_split import split_targets
from utils.timefeatures import time_features
import torch


class ClusterDataset(SafeBinaryDataset):
    def __init__(self, frame, seq_len, split, fold):
        frame = frame.sort_values(['device_id', 'date'], kind='stable').reset_index(drop=True)
        self.values = frame[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError('Invalid cluster counts')
        self.device_ids = frame.device_id.to_numpy(dtype=np.int64)
        self.dates = pd.DatetimeIndex(pd.to_datetime(frame.date, errors='raise'))
        self.seq_len = seq_len
        self.labels = (self.values > 0).any(axis=1).astype(np.float32)
        self.marks = time_features(self.dates, freq='h').T.astype(np.float32)
        indices, audits = [], []
        for device_id, group in frame.groupby('device_id', sort=True):
            dates = self.dates[group.index]
            if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
                raise ValueError(f'Invalid dates for device {device_id}')
            local = split_targets(dates, seq_len, split, fold)
            targets = local + int(group.index[0])
            indices.append(targets)
            audits.append({'device_id': int(device_id), 'windows': len(targets),
                           'positive': int(self.labels[targets].sum())})
        self.targets = np.concatenate(indices) if indices else np.array([], dtype=np.int64)
        positive = int(self.labels[self.targets].sum())
        self.summary = {'split': split, 'cv_fold': fold, 'rows': len(frame),
                        'windows': len(self.targets), 'positive': positive,
                        'negative': len(self.targets)-positive, 'devices': audits,
                        'protocol': 'paper2016_cluster_pool_v1'}
        if not len(self.targets):
            raise ValueError(f'Cluster has no {split} windows: fold={fold}, length={seq_len}')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return (*sample, torch.from_numpy(self.values[sample[-1]].copy()))
