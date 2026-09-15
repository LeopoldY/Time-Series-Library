"""Cluster all cleaned devices and train SAFE regression plus fault head end-to-end."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from data_provider.safe_binary_data import FEATURES, SafeBinaryDataset
from models.SAFE_Binary import SafeBinaryModel
from run_safe_binary import metrics
from run_safe_two_stage import TwoStageDataset, state_hash, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('dataset/fault_raw/processed_all_cleaned/four_levels'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/safe_joint_routed_training'))
    p.add_argument('--lengths', type=int, nargs='+', default=[3, 6, 12, 24])
    p.add_argument('--devices', type=int, nargs='*')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--joint-loss-weight', type=float, default=1.0,
                   help='Classification BCE weight in regression_loss + weight * BCE')
    p.add_argument('--regression-loss', choices=['mse', 'mae'], default='mse')
    p.add_argument('--head-hidden', type=int, default=16)
    p.add_argument('--head-dropout', type=float, default=.1)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=2048)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--model', choices=['Informer', 'iTransformer', 'PatchTST'])
    p.add_argument('--train', action='store_true', help='Required to start training')
    return p


def read_values(path):
    frame = pd.read_csv(path)
    required = ['date'] + FEATURES
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError('%s missing columns: %s' % (path, ', '.join(missing)))
    dates = pd.DatetimeIndex(pd.to_datetime(frame['date'], errors='raise'))
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('%s dates must be valid, unique and increasing' % path)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('%s counts must be finite and nonnegative' % path)
    return dates, values


def feature_vector(values):
    means = values.mean(axis=0)
    cvs = np.divide(values.std(axis=0), means, out=np.zeros(4), where=means > 1e-6)
    return np.concatenate(((values == 0).mean(axis=0), means, cvs))


def discover_devices(data_root, requested):
    paths = {}
    for path in sorted(data_root.glob('设备*_cleaned.csv')):
        name = path.stem[:-8]
        if name.startswith('设备') and name[2:].isdigit():
            paths[int(name[2:])] = path
    if requested:
        missing = sorted(set(requested) - set(paths))
        if missing:
            raise ValueError('Missing requested devices: ' + ', '.join(map(str, missing)))
        paths = {device: paths[device] for device in sorted(set(requested))}
    if len(paths) < 3:
        raise ValueError('At least 3 device files are required for three clusters')
    return paths


def cluster_devices(paths, output):
    features = {device: feature_vector(read_values(path)[1]) for device, path in paths.items()}
    devices = sorted(features)
    scaler = StandardScaler().fit(np.asarray([features[device] for device in devices]))
    scaled = scaler.transform(np.asarray([features[device] for device in devices]))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
    sparsity = {device: float(features[device][:4].mean()) for device in devices}
    cluster_sparsity = {cluster: float(np.mean([sparsity[device] for device, label in zip(devices, kmeans.labels_)
                                                if label == cluster])) for cluster in range(3)}
    names = {cluster: name for cluster, name in zip(sorted(cluster_sparsity, key=cluster_sparsity.get),
                                                     ('Dense', 'Intermittent', 'Sparse'))}
    models = {'Dense': 'Informer', 'Intermittent': 'iTransformer', 'Sparse': 'PatchTST'}
    rows = []
    for device, label in zip(devices, kmeans.labels_):
        category = names[int(label)]
        rows.append({'device_id': device, 'cluster': int(label), 'category': category,
                     'model': models[category], 'sparsity_mean': sparsity[device]})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / 'routing.csv', index=False)
    feature_rows = [{'device_id': device, **{('%s_%s' % (kind, name)): float(value)
                    for kind, values_part in zip(('sparsity', 'mean', 'cv'), np.split(features[device], 3))
                    for name, value in zip(FEATURES, values_part)}} for device in devices]
    pd.DataFrame(feature_rows).to_csv(output / 'device_features.csv', index=False)
    write_json(output / 'clustering.json', {'feature_order': FEATURES * 3,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'cluster_category': {str(k): v for k, v in names.items()}, 'cluster_sparsity': cluster_sparsity,
        'random_state': 42, 'n_init': 10, 'routing_version': 'safe_runtime_kmeans_v1'})
    return {row['device_id']: row for row in rows}


def run_epoch(model, loader, device, regression_criterion, classification_criterion, joint_weight,
              optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = regression_sum = classification_sum = 0.0
    count = 0
    squared, absolute = np.zeros(4), np.zeros(4)
    labels, probabilities, indices, truths, forecasts = [], [], [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, xm, dm, binary_y, target_index, counts in loader:
            x, xm, dm = x.to(device), xm.to(device), dm.to(device)
            binary_y, counts = binary_y.to(device), counts.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            forecast = model.forecast(x, xm, dm)
            logits = model.head(forecast)
            regression_loss = regression_criterion(forecast, counts)
            classification_loss = classification_criterion(logits, binary_y)
            loss = regression_loss + joint_weight * classification_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite joint loss')
            if training:
                loss.backward()
                optimizer.step()
            size = len(binary_y)
            loss_sum += loss.item() * size
            regression_sum += regression_loss.item() * size
            classification_sum += classification_loss.item() * size
            count += size
            delta = (forecast.detach().double() - counts.detach().double()).cpu().numpy()
            squared += np.square(delta).sum(axis=0)
            absolute += np.abs(delta).sum(axis=0)
            if collect:
                labels.extend(binary_y[:, 0].detach().cpu().numpy().astype(int).tolist())
                probabilities.extend(logits.sigmoid()[:, 0].detach().cpu().numpy().tolist())
                indices.extend(target_index.tolist())
                truths.append(counts.detach().cpu().numpy())
                forecasts.append(forecast.detach().cpu().numpy())
    result = {'loss': loss_sum / count, 'regression_loss': regression_sum / count,
              'classification_loss': classification_sum / count, 'windows': count,
              'mse': float(squared.sum() / (count * 4)), 'mae': float(absolute.sum() / (count * 4))}
    if collect:
        result.update({'labels': np.asarray(labels), 'probabilities': np.asarray(probabilities),
                       'indices': np.asarray(indices), 'truths': np.concatenate(truths),
                       'forecasts': np.concatenate(forecasts)})
    return result


def train_one(args, device, device_id, model_name, seq_len, path, output):
    datasets = {split: TwoStageDataset(path, seq_len, split) for split in ('train', 'val', 'test')}
    audit = {split: data.summary for split, data in datasets.items()}
    positive, negative = audit['train']['positive'], audit['train']['negative']
    if not positive or not negative:
        raise ValueError('training windows must contain both classes')
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    config.update({'device_id': str(device_id), 'model': model_name, 'seq_len': seq_len,
                   'e_layers': 3 if model_name == 'Informer' else 2, 'd_layers': 1,
                   'factor': 3, 'dropout': .1,
                   'patch_len': 1 if seq_len == 3 else seq_len // 2,
                   'stride': max(1, (1 if seq_len == 3 else seq_len // 2) // 2),
                   'training_mode': 'joint_end_to_end', 'feature_order': FEATURES,
                   'label_rule': 'any_level_next_hour', 'input_sha256': sha256_file(path),
                   'data_protocol': 'raw_counts_gap_safe_70_10_20_v1', 'data_audit': audit,
                   'pos_weight': negative / positive, 'head_parameters': args.head_hidden * 5 + 1,
                   'joint_loss_weight': args.joint_loss_weight, 'routing_version': 'safe_runtime_kmeans_v1',
                   'source_data': str(path.resolve()), 'torch_version': str(torch.__version__)})
    model_args = SimpleNamespace(**config)
    model = SafeBinaryModel(model_args).to(device)
    loaders = {split: DataLoader(data, batch_size=args.batch_size, shuffle=(split == 'train'),
                                  num_workers=args.num_workers, drop_last=False)
               for split, data in datasets.items()}
    regression_criterion = nn.MSELoss() if args.regression_loss == 'mse' else nn.L1Loss()
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_dir = output / ('device%d_%s_L%d' % (device_id, model_name, seq_len)) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'config.json', config)
    best, stale, history = float('inf'), 0, []
    checkpoint_path = run_dir / 'best_joint.pt'
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders['train'], device, regression_criterion, classification_criterion,
                          args.joint_loss_weight, optimizer)
        val = run_epoch(model, loaders['val'], device, regression_criterion, classification_criterion,
                        args.joint_loss_weight)
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
               **{('train_' + key): value for key, value in train.items() if np.isscalar(value)},
               **{('val_' + key): value for key, value in val.items() if np.isscalar(value)}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        print(json.dumps({'device': device_id, 'model': model_name, 'seq_len': seq_len, **row}), flush=True)
        if val['loss'] < best:
            best, stale = val['loss'], 0
            torch.save({'stage': 'joint', 'training_mode': 'joint_end_to_end', 'config': config,
                        'epoch': epoch, 'validation': {key: value for key, value in val.items() if np.isscalar(value)},
                        'model': model.state_dict(), 'model_sha256': state_hash(model)}, checkpoint_path)
        else:
            stale += 1
        if stale >= args.patience:
            break
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * .5 ** (epoch - 1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    test = run_epoch(model, loaders['test'], device, regression_criterion, classification_criterion,
                     args.joint_loss_weight, collect=True)
    y, probability = test['labels'], test['probabilities']
    prediction = (probability >= args.threshold).astype(int)
    classification = metrics(y, prediction, probability)
    report = {'best_epoch': checkpoint['epoch'], 'test_joint_loss': test['loss'],
              'test_regression_loss': test['regression_loss'], 'test_classification_loss': test['classification_loss'],
              'test_mse': test['mse'], 'test_mae': test['mae'], 'threshold': args.threshold,
              'classification': classification}
    write_json(run_dir / 'metrics.json', report)
    frame = {'device_id': device_id, 'target_time': datasets['test'].dates[test['indices']].astype(str)}
    for index, name in enumerate(FEATURES):
        frame[name + '_true'], frame[name + '_pred'] = test['truths'][:, index], test['forecasts'][:, index]
    frame.update({'y_true': y, 'probability': probability, 'y_pred': prediction, 'threshold': args.threshold,
                  'model': model_name, 'seq_len': seq_len, 'label_rule': 'any_level_next_hour'})
    pd.DataFrame(frame).to_csv(run_dir / 'predictions.csv', index=False)
    write_json(run_dir / 'summary.json', {'device_id': device_id, 'model': model_name,
              'seq_len': seq_len, 'training_mode': 'joint_end_to_end', 'checkpoint': str(checkpoint_path),
              'metrics': report})


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parser().parse_args()
    if not args.train:
        raise ValueError('Training is opt-in: add --train')
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.joint_loss_weight < 0:
        raise ValueError('Invalid training parameters')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    paths = discover_devices(args.data_root, args.devices)
    output = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S')
    routes = cluster_devices(paths, output)
    if args.model:
        for route in routes.values():
            route['model'] = args.model
    write_json(output / 'run_config.json', {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()})
    for device_id in sorted(paths):
        model_name = routes[device_id]['model']
        for seq_len in args.lengths:
            print('Starting device=%s model=%s seq_len=%s' % (device_id, model_name, seq_len), flush=True)
            train_one(args, device, device_id, model_name, seq_len, paths[device_id], output)


if __name__ == '__main__':
    main()