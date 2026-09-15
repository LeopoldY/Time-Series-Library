"""Runtime SAFE routing and pure evaluation with trained classification checkpoints."""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data_provider.safe_binary_data import FEATURES
from models.SAFE_Binary import SafeBinaryModel
from run_safe_binary import metrics
from utils.timefeatures import time_features


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def parse_time(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError('Invalid timestamp: ' + str(value))
    return timestamp


def device_id_from_path(path):
    match = re.fullmatch(r'设备(\d+)', path.parent.name)
    return int(match.group(1)) if match else None


def discover_data(data_root):
    grouped = {}
    invalid = []
    for path in sorted(data_root.glob('**/*.csv')):
        device_id = device_id_from_path(path)
        if device_id is None:
            invalid.append({'path': str(path), 'reason': 'parent is not 设备<id>'})
            continue
        grouped.setdefault(device_id, []).append(path)
    if not grouped:
        raise ValueError('No device CSV files found under ' + str(data_root))
    return grouped, invalid


def read_device(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = ['date'] + FEATURES
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError('%s missing columns: %s' % (path, ', '.join(missing)))
        frame = frame[required].copy()
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        for name in FEATURES:
            frame[name] = pd.to_numeric(frame[name], errors='raise')
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)
    if frame.empty:
        raise ValueError('empty device data')
    dates = pd.DatetimeIndex(frame['date'])
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('dates must be valid, unique and increasing')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    return frame, dates, values


def profile_features(values, dates, profile_end):
    mask = dates <= profile_end
    if int(mask.sum()) < 50:
        raise ValueError('profile has fewer than 50 rows')
    profile = values[mask]
    means = profile.mean(axis=0)
    stds = profile.std(axis=0)
    cvs = np.divide(stds, means, out=np.zeros(4), where=means > 1e-6)
    sparsity = (profile == 0).mean(axis=0)
    feature = np.concatenate((sparsity, means, cvs))
    return feature, int(mask.sum())


class OnlineDataset(Dataset):
    def __init__(self, dates, values, seq_len, profile_end, test_end):
        self.dates, self.values = dates, values
        self.seq_len = seq_len
        self.marks = time_features(dates, freq='h').transpose(1, 0).astype(np.float32)
        self.labels = (values > 0).any(axis=1).astype(np.float32)
        candidates = np.flatnonzero((dates > profile_end) & (dates <= test_end))
        self.targets = np.asarray([int(target) for target in candidates
                                   if target >= seq_len and self._continuous(target)], dtype=np.int64)
        if not len(self.targets):
            raise ValueError('no continuous evaluation windows')

    def _continuous(self, target):
        gaps = self.dates[target - self.seq_len:target + 1].to_series().diff().iloc[1:]
        return bool((gaps == pd.Timedelta(hours=1)).all())

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = int(self.targets[index])
        return (torch.from_numpy(self.values[target-self.seq_len:target].copy()),
                torch.from_numpy(self.marks[target-self.seq_len:target].copy()),
                torch.from_numpy(self.marks[target-1:target+1].copy()),
                torch.tensor([self.labels[target]], dtype=torch.float32), target)


def checkpoint_registry(classification_root, profile_end):
    records = []
    for path in sorted(classification_root.glob('**/stage2_head/best_head.pt')):
        record = {'checkpoint_path': str(path.resolve()), 'checkpoint_sha256': sha256_file(path),
                  'valid': False, 'reason': ''}
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            config = checkpoint.get('config', {})
            if checkpoint.get('stage') != 'head' or 'model' not in checkpoint:
                raise ValueError('not a complete head checkpoint')
            if not all(name in config for name in ('device_id', 'model', 'seq_len', 'feature_order',
                                                   'label_rule', 'data_protocol', 'threshold')):
                raise ValueError('missing compatibility fields')
            if config['feature_order'] != FEATURES or config['label_rule'] != 'any_level_next_hour':
                raise ValueError('feature or label protocol mismatch')
            record.update({
                'source_device_id': int(config['device_id']), 'model': config['model'],
                'seq_len': int(config['seq_len']), 'head_hidden': int(config['head_hidden']),
                'head_dropout': float(config['head_dropout']), 'd_model': int(config['d_model']),
                'n_heads': int(config['n_heads']), 'd_ff': int(config['d_ff']),
                'threshold': float(config['threshold']), 'feature_order': FEATURES,
                'label_rule': config['label_rule'], 'data_protocol': config['data_protocol'],
                'backbone_sha256': checkpoint.get('backbone_sha256'),
                'source_data_sha256': config.get('input_sha256'), 'shared_allowed': True,
            })
            source_root = Path(config.get('data_root', ''))
            source_file = source_root / ('设备%d_cleaned.csv' % int(config['device_id']))
            if source_file.is_file():
                source_dates = pd.to_datetime(pd.read_csv(source_file, usecols=['date'])['date'])
                train_end = int(len(source_dates) * .7)
                val_end = len(source_dates) - int(len(source_dates) * .2)
                record['training_end'] = str(source_dates.iloc[train_end - 1])
                record['selection_end'] = str(source_dates.iloc[val_end - 1])
            else:
                record['training_end'] = None
                record['selection_end'] = None
            if record['selection_end'] is None:
                raise ValueError('source training time boundary unavailable')
            if parse_time(record['selection_end']) > profile_end:
                raise ValueError('selection_end is after profile_end')
            record['selection_metric'] = 'validation_bce'
            record['selection_score'] = checkpoint.get('validation', {}).get('loss')
            record['valid'] = True
        except Exception as error:
            record['reason'] = str(error)
        records.append(record)
    if not records:
        raise ValueError('No stage2_head/best_head.pt found under ' + str(classification_root))
    return records


def route_devices(features):
    device_ids = sorted(features)
    scaler = StandardScaler().fit(np.asarray([features[i] for i in device_ids]))
    scaled = scaler.transform(np.asarray([features[i] for i in device_ids]))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
    sparsity = {device: features[device][0:4].mean() for device in device_ids}
    cluster_sparsity = {int(cluster): float(np.mean([sparsity[device] for device, label
                                                     in zip(device_ids, kmeans.labels_) if label == cluster]))
                        for cluster in range(3)}
    names = {cluster: name for cluster, name in zip(
        sorted(cluster_sparsity, key=cluster_sparsity.get), ('Dense', 'Intermittent', 'Sparse'))}
    models = {'Dense': 'Informer', 'Intermittent': 'iTransformer', 'Sparse': 'PatchTST'}
    rows = []
    for device, label in zip(device_ids, kmeans.labels_):
        category = names[int(label)]
        rows.append({'device_id': device, 'cluster': int(label), 'category': category,
                     'model': models[category], 'sparsity_mean': sparsity[device]})
    return scaler, kmeans, names, cluster_sparsity, rows


def make_plan(args, output):
    profile_end, test_end = parse_time(args.profile_end), parse_time(args.test_end)
    if test_end <= profile_end:
        raise ValueError('test-end must be after profile-end')
    grouped, invalid = discover_data(args.data_root)
    features, inventory = {}, []
    for device_id in sorted(grouped):
        try:
            frame, dates, values = read_device(grouped[device_id])
            feature, profile_rows = profile_features(values, dates, profile_end)
            features[device_id] = feature
            inventory.append({'device_id': device_id, 'files': [str(p.resolve()) for p in grouped[device_id]],
                              'rows': len(frame), 'start': str(dates[0]), 'end': str(dates[-1]),
                              'profile_rows': profile_rows, 'valid': True})
        except Exception as error:
            inventory.append({'device_id': device_id, 'files': [str(p.resolve()) for p in grouped[device_id]],
                              'valid': False, 'reason': str(error)})
    if len(features) < 3:
        raise ValueError('Fewer than 3 devices have valid profile data')
    scaler, kmeans, names, cluster_sparsity, routing = route_devices(features)
    registry = checkpoint_registry(args.classification_root, profile_end)
    valid_registry = [record for record in registry if record['valid']]
    tasks, missing = [], []
    for route in routing:
        for seq_len in args.lengths:
            candidates = [r for r in valid_registry if r['model'] == route['model'] and r['seq_len'] == seq_len]
            same = [r for r in candidates if r['source_device_id'] == route['device_id']]
            pool = same or candidates
            if same:
                chosen = sorted(pool, key=lambda r: (float(r['selection_score']), r['checkpoint_path']))
            else:
                # Shared selection is a registered deterministic fallback; do not rank
                # source-device validation scores across different data distributions.
                chosen = sorted(pool, key=lambda r: r['checkpoint_path'])
            if not chosen:
                missing.append({'device_id': route['device_id'], 'seq_len': seq_len,
                                'model': route['model'], 'reason': 'no compatible checkpoint'})
                continue
            record = chosen[0]
            transfer = 'same_device' if same else 'cross_device'
            tasks.append({'device_id': route['device_id'], 'seq_len': seq_len,
                          'model': route['model'], 'checkpoint_path': record['checkpoint_path'],
                          'checkpoint_sha256': record['checkpoint_sha256'],
                          'source_device_id': record['source_device_id'], 'transfer_mode': transfer,
                          'threshold': record['threshold']})
    protocol = {'profile_end': str(profile_end), 'test_end': str(test_end), 'lengths': args.lengths,
                'selection_policy': args.selection_policy, 'routing_version': 'safe_runtime_kmeans_v1',
                'evaluation_protocol': 'target_time_after_profile_end_and_before_test_end',
                'data_root': str(args.data_root.resolve()), 'classification_root': str(args.classification_root.resolve()),
                'created_at': datetime.now().isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    pd.DataFrame(inventory + invalid).to_csv(output / 'data_inventory.csv', index=False)
    pd.DataFrame(registry).to_csv(output / 'checkpoint_registry.csv', index=False)
    feature_rows = [{'device_id': device, **{('%s_%s' % (kind, name)): float(value)
                    for kind, values in zip(('sparsity', 'mean', 'cv'), np.split(features[device], 3))
                    for name, value in zip(FEATURES, values)}} for device in sorted(features)]
    pd.DataFrame(feature_rows).to_csv(output / 'device_features.csv', index=False)
    write_json(output / 'clustering.json', {'cluster_centers': kmeans.cluster_centers_.tolist(),
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'cluster_category': {str(k): v for k, v in names.items()}, 'cluster_sparsity': cluster_sparsity,
        'random_state': 42, 'n_init': 10})
    pd.DataFrame(routing).to_csv(output / 'routing.csv', index=False)
    plan = {'protocol': protocol, 'tasks': tasks, 'missing': missing,
            'expected_tasks': len(routing) * len(args.lengths), 'planned_tasks': len(tasks)}
    write_json(output / 'plan.json', plan)
    pd.DataFrame([{'device_id': t['device_id'], 'seq_len': t['seq_len'], 'status': 'planned',
                   'checkpoint_path': t['checkpoint_path'], 'transfer_mode': t['transfer_mode']}
                  for t in tasks] + [{'device_id': m['device_id'], 'seq_len': m['seq_len'],
                                      'status': 'missing', 'reason': m['reason']} for m in missing]
                 ).to_csv(output / 'coverage.csv', index=False)
    print(json.dumps({'output': str(output), 'devices': len(routing),
                      'expected_tasks': plan['expected_tasks'], 'planned_tasks': len(tasks),
                      'missing': len(missing)}, ensure_ascii=False, indent=2))
    if missing and not args.allow_partial:
        raise RuntimeError('Task coverage incomplete; use --allow-partial only for diagnosis')


def evaluate_plan(plan_path, device_name, allow_partial):
    plan_root = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    results, coverage, prediction_frames = [], [], []
    grouped, _ = discover_data(Path(plan['protocol']['data_root']))
    device_data = {}
    for device_id in sorted({task['device_id'] for task in plan['tasks']}):
        frame, dates, values = read_device(grouped[device_id])
        device_data[device_id] = (frame, dates, values)
    for task in plan['tasks']:
        row = {'device_id': task['device_id'], 'seq_len': task['seq_len'], 'status': 'failed',
               'checkpoint_path': task['checkpoint_path'], 'transfer_mode': task['transfer_mode']}
        try:
            frame, dates, values = device_data[task['device_id']]
            dataset = OnlineDataset(dates, values, task['seq_len'], parse_time(plan['protocol']['profile_end']),
                                    parse_time(plan['protocol']['test_end']))
            checkpoint_path = Path(task['checkpoint_path'])
            if sha256_file(checkpoint_path) != task['checkpoint_sha256']:
                raise ValueError('checkpoint SHA256 changed since planning')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            config = SimpleNamespace(**checkpoint['config'])
            model = SafeBinaryModel(config).to(device)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.freeze_backbone()
            before = state_hash(model.backbone)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            probabilities, labels, indices = [], [], []
            with torch.no_grad():
                for x, xm, dm, target, target_indices in loader:
                    probabilities.extend(model(x.to(device), xm.to(device), dm.to(device)).sigmoid()[:, 0].cpu().tolist())
                    labels.extend(target[:, 0].numpy().astype(int).tolist())
                    indices.extend(target_indices.tolist())
            after = state_hash(model.backbone)
            if before != after or before != checkpoint.get('backbone_sha256'):
                raise ValueError('backbone changed or checkpoint hash mismatch')
            probability = np.asarray(probabilities)
            truth = np.asarray(labels, dtype=int)
            prediction = (probability >= task['threshold']).astype(int)
            report = metrics(truth, prediction, probability)
            report.update({'device_id': task['device_id'], 'seq_len': task['seq_len'],
                           'model': task['model'], 'source_device_id': task['source_device_id'],
                           'transfer_mode': task['transfer_mode'], 'samples': len(truth),
                           'target_start': str(dates[indices[0]]), 'target_end': str(dates[indices[-1]]),
                           'checkpoint_sha256': task['checkpoint_sha256']})
            out = plan_root / 'predictions' / ('device%d' % task['device_id']) / ('L%d' % task['seq_len'])
            out.mkdir(parents=True, exist_ok=True)
            prediction_frame = pd.DataFrame({'device_id': task['device_id'], 'target_time': dates[indices].astype(str),
                          'y_true': truth, 'probability': probability, 'y_pred': prediction,
                          'threshold': task['threshold'], 'source_device_id': task['source_device_id'],
                          'transfer_mode': task['transfer_mode'], 'checkpoint_sha256': task['checkpoint_sha256'],
                          'routing_version': plan['protocol']['routing_version'],
                          'evaluation_protocol': plan['protocol']['evaluation_protocol']})
            prediction_frame.to_csv(out / 'predictions.csv', index=False)
            prediction_frames.append((task, prediction_frame))
            results.append(report)
            row.update({'status': 'success', 'samples': len(truth), 'checkpoint_sha256': task['checkpoint_sha256']})
        except Exception as error:
            row['reason'] = str(error)
        coverage.append(row)
    metric_columns = ['device_id', 'seq_len', 'model', 'source_device_id', 'transfer_mode',
                      'samples', 'target_start', 'target_end', 'checkpoint_sha256',
                      'tn', 'fp', 'fn', 'tp', 'positive', 'negative', 'accuracy',
                      'precision', 'recall', 'f1', 'false_positive_rate',
                      'average_precision', 'roc_auc']
    metrics_frame = pd.DataFrame(results, columns=metric_columns)
    metrics_frame.to_csv(plan_root / 'all_metrics.csv', index=False)
    if results:
        metrics_frame.groupby('seq_len', as_index=False).mean(numeric_only=True).to_csv(
            plan_root / 'by_length.csv', index=False)
    else:
        pd.DataFrame(columns=['seq_len']).to_csv(plan_root / 'by_length.csv', index=False)
    common_rows = []
    for seq_len in sorted({task['seq_len'] for task, _ in prediction_frames}):
        frames = [(task, frame) for task, frame in prediction_frames if task['seq_len'] == seq_len]
        common_times = set.intersection(*(set(frame['target_time']) for _, frame in frames)) if frames else set()
        per_device = []
        for task, frame in frames:
            common = frame[frame['target_time'].isin(common_times)]
            if len(common):
                per_device.append(metrics(common.y_true.to_numpy(), common.y_pred.to_numpy(),
                                          common.probability.to_numpy()))
        numeric = {key: float(np.mean([row[key] for row in per_device if row.get(key) is not None]))
                   for key in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate',
                               'average_precision', 'roc_auc')
                   if any(row.get(key) is not None for row in per_device)}
        common_rows.append({'seq_len': seq_len, 'common_target_start': min(common_times) if common_times else None,
                            'common_target_end': max(common_times) if common_times else None,
                            'common_samples_per_device': len(common_times),
                            'effective_devices': len(per_device), **numeric})
    pd.DataFrame(common_rows, columns=['seq_len', 'common_target_start', 'common_target_end',
                                       'common_samples_per_device', 'effective_devices',
                                       'accuracy', 'precision', 'recall', 'f1',
                                       'false_positive_rate', 'average_precision', 'roc_auc']).to_csv(
        plan_root / 'common_time_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(plan_root / 'coverage.csv', index=False)
    success = sum(row['status'] == 'success' for row in coverage)
    complete = success == plan['expected_tasks'] and not plan.get('missing')
    write_json(plan_root / 'completion.json', {'complete': complete, 'expected_tasks': plan['expected_tasks'],
              'success': success, 'failed': len(coverage) - success, 'missing': len(plan.get('missing', [])),
              'pure_inference': True, 'optimizer_created': False, 'backward_called': False})
    report = ['# SAFE全设备纯验证报告', '',
              '- 设备数：%d' % len({task['device_id'] for task in plan['tasks']}),
              '- 预期任务：%d' % plan['expected_tasks'], '- 成功任务：%d' % success,
              '- 失败任务：%d' % (len(coverage) - success),
              '- 完整完成：%s' % ('true' if complete else 'false'),
              '- 纯推理：true（未创建优化器，未调用反向传播）', '',
              '详细结果见 `all_metrics.csv`、`by_length.csv`、`common_time_metrics.csv` 和 `coverage.csv`。']
    (plan_root / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if not complete and not allow_partial:
        raise RuntimeError('Evaluation incomplete; see coverage.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('dataset/fault'))
    parser.add_argument('--classification-root', type=Path, default=Path('outputs'))
    parser.add_argument('--lengths', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--profile-end')
    parser.add_argument('--test-end')
    parser.add_argument('--selection-policy', choices=['same_device_then_shared'], default='same_device_then_shared')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/safe_all_devices_evaluation'))
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.plan_only == bool(args.plan) or args.plan_only == args.evaluate_only:
        raise ValueError('Use exactly one of --plan-only or --evaluate-only --plan PATH')
    if args.plan_only:
        if not args.profile_end or not args.test_end:
            raise ValueError('--profile-end and --test-end are required for planning')
        output = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S')
        make_plan(args, output)
    else:
        evaluate_plan(args.plan, args.device, args.allow_partial)


if __name__ == '__main__':
    main()