"""Reproduce documented paper filters, retaining existing empty-column/zero-run rules."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ['提示', '次要', '重要', '紧急']
FEATURES = [f'w_level{i}' for i in range(1, 5)]
THRESHOLDS = dict(zip(LEVELS, [900, 240, 180, 120]))


def clean_events(raw):
    frame = raw.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all').copy()
    required = ['告警源', '级别', '名称', '告警类型', '发生时间', 'Repair_Interval']
    if not set(required) <= set(frame) or frame[required[:-1]].isna().any().any():
        raise ValueError('Missing required event fields')
    if not frame['级别'].isin(LEVELS).all():
        raise ValueError('Unknown severity')
    frame['_time'] = pd.to_datetime(frame['发生时间'], format='%m/%d/%Y %H:%M:%S', errors='raise')
    repair = pd.to_numeric(frame['Repair_Interval'], errors='coerce')
    derived = frame['告警类型'].eq('衍生告警')
    flash = ~derived & repair.ge(0) & repair.lt(22)
    kept = frame.loc[~derived & ~flash].sort_values('_time', kind='stable')
    last = {}; keep = []
    for idx, row in kept.iterrows():
        key = tuple(row[k] for k in ['告警源', '级别', '名称', '告警类型'])
        previous = last.get(key)
        if previous is None or (row['_time'] - previous).total_seconds() >= THRESHOLDS[row['级别']]:
            keep.append(idx)
            last[key] = row['_time']
    result = kept.loc[keep]
    audit = dict(raw_events=len(raw), empty_columns_removed=len(raw.columns)-len(frame.columns)+1,
                 derived_removed=int(derived.sum()), flash_removed=int(flash.sum()),
                 duplicate_removed=len(kept)-len(result), retained_events=len(result),
                 unknown_duration_retained=int((~np.isfinite(repair) | repair.lt(0)).loc[keep].sum()))
    assert sum(audit[k] for k in ['derived_removed','flash_removed','duplicate_removed','retained_events']) == len(raw)
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'dataset/fault_raw/raw_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'dataset/fault_paper2016')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'docs/paper2016')
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob('*.csv'), key=lambda p: int(p.name.split('_')[0][2:]))
    if not paths:
        raise ValueError('No source files')
    devices = []
    for path in paths:
        raw = pd.read_csv(path)
        name = path.name.split('_')[0]
        if not raw['告警源'].eq(name).all():
            raise ValueError(f'Device mismatch: {path}')
        times = pd.to_datetime(raw['发生时间'], format='%m/%d/%Y %H:%M:%S', errors='raise')
        events, audit = clean_events(raw)
        devices.append((path, name, times, events, audit))
    start = min(t.min() for _,_,t,_,_ in devices)
    end = max(t.max() for _,_,t,_,_ in devices)
    dates = pd.date_range(start + pd.Timedelta(minutes=30), periods=int((end-start)//pd.Timedelta(hours=1))+1, freq='h')
    for folder in ['events','four_levels','folds']:
        (args.output_dir/folder).mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    audits=[]
    for path,name,times,events,audit in devices:
        events.drop(columns='_time').to_csv(args.output_dir/'events'/path.name,index=False)
        counts=np.zeros((len(dates),4),dtype=np.int64)
        bins=((events['_time']-start)//pd.Timedelta(hours=1)).to_numpy(dtype=int)
        levels=events['级别'].map(dict(zip(LEVELS,range(4)))).to_numpy(dtype=int)
        np.add.at(counts,(bins,levels),1)
        frame=pd.DataFrame(counts,columns=FEATURES)
        frame.insert(0,'date',dates)
        zero=frame[FEATURES].eq(0).all(axis=1)
        groups=zero.ne(zero.shift()).cumsum()
        remove=zero & zero.groupby(groups).transform('size').ge(336)
        frame=frame.loc[~remove].reset_index(drop=True)
        target=args.output_dir/'four_levels'/f'{name}_cleaned.csv'
        frame.to_csv(target,index=False)
        saved=pd.read_csv(target)
        assert int(saved[FEATURES].sum().sum())==len(events)
        assert len(saved)==len(frame)
        fold=np.minimum(np.arange(len(frame))*10//max(1,len(frame)),9)
        pd.DataFrame({'row':np.arange(len(frame)),'date':frame.date,'fold':fold}).to_csv(args.output_dir/'folds'/f'{name}.csv',index=False)
        audits.append(dict(device=name,**audit,removed_zero_hours=int(remove.sum()),hourly_rows=len(frame),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    pd.DataFrame(audits).to_csv(args.report_dir/'cleaning_audit.csv',index=False)
    summary=dict(devices=len(devices),**{key:sum(a[key] for a in audits) for key in ['raw_events','derived_removed','flash_removed','duplicate_removed','retained_events','removed_zero_hours','hourly_rows','unknown_duration_retained']},
                 protocol='paper2016_blocked10_v1',first_label=str(dates[0]),last_label=str(dates[-1]))
    (args.report_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
