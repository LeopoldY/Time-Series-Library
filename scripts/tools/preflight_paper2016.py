"""Audit every device/length/fold before training, without requiring PyTorch."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from data_provider.paper2016_split import split_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'dataset/fault_paper2016/four_levels')
    p.add_argument('--output',type=Path,default=ROOT/'docs/paper2016/split_audit.csv')
    args=p.parse_args(); rows=[]
    for path in sorted(args.data_root.glob('设备*_cleaned.csv')):
        frame=pd.read_csv(path); dates=pd.to_datetime(frame.date)
        labels=frame.iloc[:,1:5].gt(0).any(axis=1).to_numpy()
        for fold in range(10):
            for length in [3,6,12,24]:
                record=dict(device=path.stem[:-8],fold=fold,seq_len=length)
                targets={s:split_targets(dates,length,s,fold) for s in ['train','val','test']}
                for split,indices in targets.items():
                    record[split+'_windows']=len(indices)
                    record[split+'_positive']=int(labels[indices].sum())
                reasons=[]
                if any(not len(t) for t in targets.values()): reasons.append('empty_split')
                train=labels[targets['train']]
                if len(train) and (train.all() or not train.any()): reasons.append('single_class_train')
                record['status']=';'.join(reasons) or 'ready'
                rows.append(record)
    if not rows: raise ValueError('No input devices')
    result=pd.DataFrame(rows); args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.status.value_counts().to_string())


if __name__=='__main__': main()
