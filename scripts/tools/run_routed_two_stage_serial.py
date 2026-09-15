"""Preflight all routed jobs, then optionally run device-wise serial two-stage training."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_provider.safe_binary_data import SafeBinaryDataset


def build_plan(data_root, routes, lengths, seeds, device, output_root):
    import csv
    with routes.open(encoding='utf-8-sig', newline='') as stream:
        assignments = list(csv.DictReader(stream))
    names = [r['Device_Name'] for r in assignments]
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate routing table')
    expected = {p.stem[:-len('_cleaned')] for p in data_root.glob('设备*_cleaned.csv')}
    if set(names) != expected:
        raise ValueError('Route/data coverage mismatch')
    jobs = []
    for route in sorted(assignments, key=lambda r: int(r['Device_ID'])):
        name, model = route['Device_Name'], route['Model']
        if model not in ('Informer', 'iTransformer', 'PatchTST') or name != '设备'+route['Device_ID']:
            raise ValueError('Invalid route: '+name)
        path = data_root/(name+'_cleaned.csv')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != route['Input_SHA256']:
            raise ValueError('Input differs from routed dataset: '+name)
        for length in lengths:
            audit = {s: SafeBinaryDataset(path, length, s).summary for s in ['train', 'val', 'test']}
            has_classes = audit['train']['positive'] > 0 and audit['train']['negative'] > 0
            for seed in seeds:
                options = ['--device-id', route['Device_ID'], '--model', model, '--seq-len', str(length),
                    '--stage', 'both' if has_classes else 'regression', '--regression-loss', 'mse',
                    '--regression-epochs', '100', '--epochs', '100', '--patience', '3',
                    '--learning-rate', '0.0001', '--head-learning-rate', '0.001',
                    '--batch-size', '16' if model == 'PatchTST' else '32', '--num-workers', '0',
                    '--seed', str(seed), '--d-model', '512', '--d-ff', '2048', '--n-heads', '8',
                    '--head-hidden', '16', '--head-dropout', '0.1', '--head-type', 'history',
                    '--head-loss', 'asymmetric', '--asl-gamma-neg', '4', '--asl-gamma-pos', '0',
                    '--asl-clip', '0.05', '--head-selection', 'ap', '--threshold-policy', 'val_fbeta',
                    '--threshold-beta', '1', '--threshold', '0.5', '--device', device]
                jobs.append(dict(id=f'device{route["Device_ID"]}_{model}_L{length}_seed{seed}',
                    device=name, model=model, seq_len=length, seed=seed, input_sha256=digest,
                    classification_status='scheduled' if has_classes else 'blocked_single_class_train',
                    audit=audit, options=options))
    return dict(protocol='raw_counts_gap_safe_70_10_20_v1', label_rule='any_level_next_hour',
        routing_sha256=hashlib.sha256(routes.read_bytes()).hexdigest(), devices=len(names),
        regression_jobs=len(jobs), classification_jobs=sum(j['classification_status']=='scheduled' for j in jobs),
        execution='serial: each device/length/seed regression then classification, then next job',
        data_root=str(data_root), output_root=str(output_root), jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'dataset/fault_all_selected_cleaned')
    parser.add_argument('--routes', type=Path, default=ROOT/'docs/safe_all_selected_cleaned_clustering/device_model_assignments.csv')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/safe_all_serial')
    parser.add_argument('--plan-path', type=Path, default=ROOT/'outputs/safe_all_serial/plan.json')
    parser.add_argument('--lengths', type=int, nargs='+', default=[24], choices=[3,6,12,24])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true', help='Without this option only preflight/write a plan')
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate lengths or seeds')
    if any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError('Seeds must lie in [0, 2**32)')
    plan = build_plan(args.data_root, args.routes, args.lengths, args.seeds, args.device, args.output_root)
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in plan.items() if k!='jobs'}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print('Preflight complete; no training launched.')
        return
    # Exclusive experiment directory prevents silently rerunning successful jobs.
    execution = args.output_root/'execution'
    execution.mkdir(parents=True, exist_ok=False)
    for job in plan['jobs']:
        # Recheck before launch in case data changed after preflight.
        path = args.data_root/(job['device']+'_cleaned.csv')
        if hashlib.sha256(path.read_bytes()).hexdigest() != job['input_sha256']:
            raise RuntimeError('Data changed after preflight: '+job['device'])
        command = [sys.executable, str(ROOT/'run_safe_two_stage.py'), *job['options'],
            '--data-root', str(args.data_root.resolve()), '--output-root', str((execution/job['id']).resolve()), '--train']
        state_path = execution/(job['id']+'.json')
        state = dict(id=job['id'], status='running', classification_status=job['classification_status'], command=command)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('Running '+job['id'], flush=True)
        with (execution/(job['id']+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Job failed; later jobs not started: '+job['id'])
    print('All scheduled jobs finished. Check blocked classification statuses separately.')


if __name__ == '__main__':
    main()
