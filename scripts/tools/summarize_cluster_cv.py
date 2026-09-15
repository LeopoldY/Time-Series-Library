"""Report equal-weight fold means; never label incomplete runs as ten-fold results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ['test_mse', 'test_mae', 'accuracy', 'precision', 'recall', 'f1',
           'balanced_accuracy', 'false_positive_rate', 'average_precision', 'roc_auc']


def summarize(root):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    jobs = plan['jobs']
    identities = [(j['fold'], j['model'], j['seq_len']) for j in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate jobs in plan')
    rows = []
    for job in jobs:
        marker = root / 'jobs' / job['id'] / 'complete.json'
        if not marker.exists():
            continue
        complete = json.loads(marker.read_text())
        path = root / complete['run'] / 'summary.json'
        key = str(path.relative_to(root))
        if hashlib.sha256(path.read_bytes()).hexdigest() != complete['artifacts'][key]:
            raise ValueError(f'Modified summary: {job["id"]}')
        summary = json.loads(path.read_text())
        if summary['model'] != job['model'] or summary['seq_len'] != job['seq_len']:
            raise ValueError('Completed job identity mismatch')
        report = summary['metrics']
        values = {**report, **report['classification']}
        positive, negative = values.get('positive', 0), values.get('negative', 0)
        values['balanced_accuracy'] = ((values['tp'] / positive + values['tn'] / negative) / 2
                                       if positive and negative else None)
        rows.append({'fold': job['fold'], 'model': job['model'], 'seq_len': job['seq_len'],
                     'test_windows': job['audit']['test']['windows'],
                     **{m: values.get(m) for m in METRICS}})
    columns = ['fold', 'model', 'seq_len', 'test_windows'] + METRICS
    pd.DataFrame(rows, columns=columns).to_csv(root / 'fold_metrics.csv', index=False)
    averages = []
    for model, length in sorted(set((j['model'], j['seq_len']) for j in jobs)):
        records = [r for r in rows if r['model'] == model and r['seq_len'] == length]
        folds = sorted(r['fold'] for r in records)
        complete_ten = folds == list(range(10))
        row = {'model': model, 'seq_len': length, 'completed_folds': len(folds),
               'ten_fold_complete': complete_ten}
        for metric in METRICS:
            available = [r[metric] for r in records if r[metric] is not None and np.isfinite(r[metric])]
            row[metric + '_valid_folds'] = len(available)
            # Undefined AUC/AP in single-class folds remains undefined for the
            # ten-fold mean. Do not silently report an average over fewer folds.
            valid = complete_ten and len(available) == 10
            row[metric + '_mean'] = float(np.mean(available)) if valid else None
            row[metric + '_std'] = float(np.std(available, ddof=1)) if valid else None
        averages.append(row)
    pd.DataFrame(averages).to_csv(root / 'cv_summary.csv', index=False)
    status = {'planned_jobs': len(jobs), 'completed_jobs': len(rows),
              'all_planned_jobs_complete': len(rows) == len(jobs),
              'full_120_job_run': len(jobs) == 120 and len(averages) == 12
                  and set(j['seq_len'] for j in jobs) == {3, 6, 12, 24}
                  and all(r['ten_fold_complete'] for r in averages)}
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    lines = ['# 聚类聚合训练交叉验证结果', '',
             f"完成 {len(rows)}/{len(jobs)} 组；均值为十折等权平均，标准差使用 ddof=1。", '',
             '各模型按历史长度分别报告，保留12行；不将不同长度混为同一实验。', '',
             '| 模型 | 历史长度 | 完成折数 | MSE | MAE | Precision | Recall | F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in averages:
        cells = []
        for metric in ['test_mse', 'test_mae', 'precision', 'recall', 'f1']:
            value = row[metric + '_mean']
            cells.append(f"{value:.6f} ± {row[metric + '_std']:.6f}" if value is not None else '未完成/未定义')
        lines.append(f"| {row['model']} | {row['seq_len']} | {row['completed_folds']} | " + ' | '.join(cells) + ' |')
    lines += ['', '完整指标见 cv_summary.csv，各折结果见 fold_metrics.csv。缺折不计算十折均值；',
              '单类测试折导致AUC等指标未定义时，报告有效折数并留空该指标的十折均值。',
              '每折先合并本簇所有设备测试样本计算指标，再对十折等权平均。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    print(json.dumps(summarize(parser.parse_args().run_dir), indent=2))
