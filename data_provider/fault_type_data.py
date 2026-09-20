"""Causal hourly onset prediction from raw alarm names, preserving quiet hours."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features

LEVELS = {'提示': 0, '次要': 1, '重要': 2, '紧急': 3}


def prepare(path):
    path = Path(path)
    raw = pd.read_csv(path, encoding='utf-8-sig')
    required = ['ID', '名称', '级别', '告警源', '发生时间']
    if raw[required].isna().any().any() or not len(raw):
        raise ValueError('Missing required alarm fields')
    raw['名称'] = raw['名称'].astype(str).str.strip()
    if raw['名称'].eq('').any():
        raise ValueError('Empty fault name')
    expected = path.name.split('_')[0]
    if not raw['告警源'].eq(expected).all():
        raise ValueError('Device source mismatch')
    # An ID is an event identity. Conflicting repeated identities must be investigated.
    if raw.groupby('ID')[required[1:]].nunique().gt(1).any().any():
        raise ValueError('Conflicting duplicate event IDs')
    original = len(raw)
    raw = raw.drop_duplicates('ID').copy()
    duplicates_removed = original-len(raw)
    times = pd.to_datetime(raw['发生时间'], format='%m/%d/%Y %H:%M:%S', errors='raise')
    levels = raw['级别'].map(LEVELS)
    if levels.isna().any():
        raise ValueError('Unknown alarm severity')
    # Exclude the incomplete last hour, whose absence of later events is not observed.
    start, end = times.min().ceil('h'), times.max().floor('h')
    dates = pd.date_range(start, end-pd.Timedelta(hours=1), freq='h')
    if len(dates) < 100:
        raise ValueError('Insufficient observation span')
    keep = (times >= start) & (times < end)
    raw, times, levels = raw.loc[keep], times.loc[keep], levels.loc[keep]
    bins = ((times - start) // pd.Timedelta(hours=1)).to_numpy(dtype=int)
    train_end, val_end = int(len(dates)*.7), int(len(dates)*.8)
    # Full catalog is for audit/evaluation only. Model vocabulary uses train events only.
    catalog = sorted(raw['名称'].unique().tolist())
    names = sorted(raw.loc[bins < train_end, '名称'].unique().tolist())
    if not names:
        raise ValueError('No training fault types')
    counts = np.zeros((len(dates), len(catalog)), dtype=np.float32)
    np.add.at(counts, (bins, pd.Categorical(raw['名称'], categories=catalog).codes), 1)
    severity = np.zeros((len(dates), 4), dtype=np.float32)
    np.add.at(severity, (bins, levels.to_numpy(dtype=int)), 1)
    columns = [catalog.index(name) for name in names]
    # History includes type counts and severity; unknown future names affect severity only.
    values = np.log1p(np.concatenate([severity, counts[:, columns]], axis=1))
    mean, scale = values[:train_end].mean(0), values[:train_end].std(0)
    scale[scale < 1e-6] = 1
    audit = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 raw_events=original, duplicate_ids_removed=duplicates_removed,
                 partial_boundary_events_removed=int((~keep).sum()), retained_events=len(raw),
                 hours=len(dates), start=str(start), end_exclusive=str(end),
                 train_end=train_end, val_end=val_end, names=names, catalog=catalog,
                 unseen_types=sorted(set(catalog)-set(names)), mean=mean.tolist(), scale=scale.tolist(),
                 type_event_counts={name: int(counts[:, i].sum()) for i, name in enumerate(catalog)},
                 protocol='fault_types_hourly_onset_v1', quiet_hours_retained=True)
    return dict(values=((values-mean)/scale).astype(np.float32), counts=counts, severity=severity,
                labels=(counts[:, columns] > 0).astype(np.float32),
                binary=(counts.sum(1)>0).astype(np.float32), dates=dates,
                marks=time_features(dates, freq='h').T.astype(np.float32), audit=audit)


class FaultTypeDataset(Dataset):
    def __init__(self, data, seq_len, split):
        if seq_len < 1 or split not in ('train', 'val', 'test'):
            raise ValueError('Invalid window/split')
        self.data, self.seq_len = data, seq_len
        n = len(data['dates'])
        a, b = data['audit']['train_end'], data['audit']['val_end']
        lo, hi = {'train': (0, a), 'val': (a, b), 'test': (b, n)}[split]
        # Historical context may precede the split; target hours are strictly disjoint.
        self.targets = np.arange(max(lo, seq_len), hi)
        if not len(self.targets):
            raise ValueError('Empty split')
        y = data['labels'][self.targets]
        self.summary = dict(windows=len(y), binary_positive=int(data['binary'][self.targets].sum()),
                            type_positive=y.sum(0).astype(int).tolist(),
                            first_target=str(data['dates'][self.targets[0]]),
                            last_target=str(data['dates'][self.targets[-1]]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        t = self.targets[index]
        d, length = self.data, self.seq_len
        return (torch.from_numpy(d['values'][t-length:t].copy()),
                torch.from_numpy(d['marks'][t-length:t].copy()),
                torch.from_numpy(d['marks'][t-1:t+1].copy()),
                torch.from_numpy(np.r_[d['binary'][t], d['labels'][t]].astype(np.float32)), int(t))
