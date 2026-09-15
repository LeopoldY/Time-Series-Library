"""Train one pooled model per cluster/length/fold, with resumable 120-job plans."""
import copy
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_provider.safe_cluster_data import ClusterDataset
from run_safe_joint_routed_training import (
    parser as base_parser, discover_devices, cluster_devices, read_values,
    sha256_file, train_one, write_json,
)
from scripts.tools.summarize_cluster_cv import summarize

MODELS = ('Informer', 'iTransformer', 'PatchTST')


def parser():
    p = base_parser()
    p.description = __doc__
    p.set_defaults(data_root=Path('dataset/fault_paper2016/four_levels'),
                   output_root=Path('outputs/safe_cluster_cv'))
    p.add_argument('--run-dir', type=Path, help='Existing plan resumes only unfinished jobs')
    p.add_argument('--folds', type=int, nargs='+', default=list(range(10)))
    return p


def settings(args):
    ignored = {'train', 'run_dir', 'output_root'}
    return {k: str(v.resolve()) if isinstance(v, Path) else v
            for k, v in vars(args).items() if k not in ignored}


def build_plan(args, root):
    paths = discover_devices(args.data_root, args.devices)
    inputs = {str(d): sha256_file(p) for d, p in paths.items()}
    plan_path = root / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['settings'] != settings(args) or plan['inputs'] != inputs:
            raise ValueError('Resume settings or source hashes differ; use a new run directory')
        for job in plan['jobs']:
            if sha256_file(root / job['dataset']) != job['dataset_sha256']:
                raise ValueError('Aggregated dataset changed')
        return plan
    if root.exists() and any(root.iterdir()):
        raise ValueError('Nonempty run directory without completed plan; use a new run directory')
    root.mkdir(parents=True, exist_ok=True)
    frames = {}
    for device_id, path in paths.items():
        dates, values = read_values(path)
        if not len(values):
            raise ValueError(f'Empty device {device_id}; cannot assign a training cluster')
        frame = pd.read_csv(path)
        frame.insert(0, 'device_id', device_id)
        frames[device_id] = frame
    jobs, audit_rows = [], []
    for fold in args.folds:
        routes = cluster_devices(paths, root / f'fold{fold}' / 'routing', fold)
        for model in MODELS:
            members = sorted(d for d, route in routes.items() if route['model'] == model)
            if not members:
                raise ValueError(f'Empty cluster {model}, fold {fold}')
            # Materialize ONE pooled hourly dataset per cluster and fold. Device ID
            # is metadata, never a feature; windows are constructed inside devices.
            dataset = Path(f'fold{fold}/datasets/{model}.csv')
            (root / dataset).parent.mkdir(parents=True, exist_ok=True)
            combined = pd.concat([frames[d] for d in members], ignore_index=True)
            combined.to_csv(root / dataset, index=False)
            digest = sha256_file(root / dataset)
            for length in args.lengths:
                splits = {s: ClusterDataset(combined, length, s, fold) for s in ('train', 'val', 'test')}
                audit = {s: ds.summary for s, ds in splits.items()}
                if not audit['train']['positive'] or not audit['train']['negative']:
                    raise ValueError(f'Pooled train set has one class: {model}, fold={fold}, L={length}')
                job = {'id': f'fold{fold}_{model}_L{length}', 'fold': fold, 'model': model,
                       'seq_len': length, 'devices': members, 'dataset': str(dataset),
                       'dataset_sha256': digest, 'audit': audit}
                jobs.append(job)
                audit_rows.append({'fold': fold, 'model': model, 'seq_len': length,
                                   'devices': len(members), **{f'{s}_{k}': audit[s][k]
                                   for s in splits for k in ('windows', 'positive', 'negative')}})
        print(f'Prepared fold {fold}: 3 pooled datasets', flush=True)
    plan = {'protocol': 'paper2016_cluster_pool_v1', 'settings': settings(args),
            'inputs': inputs, 'jobs': jobs}
    pd.DataFrame(audit_rows).to_csv(root / 'preflight.csv', index=False)
    write_json(plan_path, plan)
    return plan


def main():
    args = parser().parse_args()
    if args.model is not None or args.cv_fold is not None:
        raise ValueError('Cluster CV uses all three routed models; use --folds to select folds')
    if (not args.folds or len(set(args.folds)) != len(args.folds)
            or any(f not in range(10) for f in args.folds)
            or not args.lengths or len(set(args.lengths)) != len(args.lengths)
            or any(n < 3 for n in args.lengths)):
        raise ValueError('Invalid or duplicate folds/lengths')
    if (args.epochs < 1 or args.patience < 1 or args.batch_size < 1
            or args.learning_rate <= 0 or args.joint_loss_weight < 0
            or not 0 <= args.threshold <= 1):
        raise ValueError('Invalid training parameters')
    root = args.run_dir or args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = root.resolve()
    print(f'Run directory: {root}', flush=True)
    plan = build_plan(args, root)
    print(f"Preflight passed: {len(plan['jobs'])} pooled training jobs", flush=True)
    if not args.train:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    for job in plan['jobs']:
        job_root = root / 'jobs' / job['id']
        complete = job_root / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            for filename, digest in saved['artifacts'].items():
                if sha256_file(root / filename) != digest:
                    raise ValueError(f'Completed artifact changed: {filename}')
            print('Already complete:', job['id'], flush=True)
            continue
        job_root.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(root / job['dataset'])
        splits = {s: ClusterDataset(frame, job['seq_len'], s, job['fold']) for s in ('train', 'val', 'test')}
        # Independent deterministic initialization; interrupted jobs restart fully.
        seed = args.seed + job['fold'] * 1000 + job['seq_len'] * 10 + MODELS.index(job['model'])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        options = copy.copy(args)
        options.cv_fold, options.seed = job['fold'], seed
        run = train_one(options, device, 'cluster', job['model'], job['seq_len'],
                        root / job['dataset'], job_root, datasets=splits,
                        metadata={'data_protocol': 'paper2016_cluster_pool_v1',
                                  'cluster_devices': job['devices'], 'cv_fold': job['fold'],
                                  'pool_weighting': 'equal_per_window', 'job_id': job['id']})
        artifacts = {str((run / name).relative_to(root)): sha256_file(run / name)
                     for name in ('summary.json', 'metrics.json', 'predictions.csv', 'best_joint.pt', 'config.json')}
        marker = job_root / 'complete.tmp'
        write_json(marker, {'run': str(run.relative_to(root)), 'artifacts': artifacts})
        marker.replace(complete)
        summarize(root)
    report = summarize(root)
    if not report['all_planned_jobs_complete']:
        raise RuntimeError('Some planned jobs have not completed')
    print(f"Finished: {root / 'cv_summary.csv'}", flush=True)


if __name__ == '__main__':
    main()
