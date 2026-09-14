"""Collect completed stage metrics without starting or changing any training."""
import argparse
import csv
import json
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[4]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=project/'outputs/safe_two_stage_cleaned')
    args = p.parse_args()
    rows = []
    for path in sorted(args.output_root.glob('*/*/summary.json')):
        summary = json.loads(path.read_text())
        config = json.loads((path.parent/'config.json').read_text())
        row = {k: summary[k] for k in ['device_id', 'model', 'seq_len']}
        row.update(run_dir=str(path.parent), regression_loss=config['regression_loss'],
                   regression_mse=summary.get('regression', {}).get('mse'),
                   regression_mae=summary.get('regression', {}).get('mae'))
        classification = summary.get('classification', {}).get('model', {})
        row.update({k: classification.get(k) for k in ['precision', 'recall', 'f1', 'average_precision', 'roc_auc']})
        rows.append(row)
    if not rows:
        print('No completed stage summaries found.')
        return
    target = args.output_root/'comparison.csv'
    with target.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} runs: {target}')


if __name__ == '__main__':
    main()
