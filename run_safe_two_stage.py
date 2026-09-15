"""Forecast pretraining, then a frozen-backbone binary head. --train opts in."""
import hashlib
import math
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_provider.safe_binary_data import SafeBinaryDataset, FEATURES
from models.SAFE_Binary import SafeBinaryModel
from models.SAFE_Imbalance import AsymmetricBinaryLoss, select_threshold
from run_safe_binary import ROOT, configure, metrics, parser as binary_parser, write_json


class TwoStageDataset(SafeBinaryDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        target = torch.from_numpy(self.values[sample[-1]].copy())
        return (*sample, target)


def parser():
    p = binary_parser()
    p.description = __doc__
    p.set_defaults(output_root=ROOT/'outputs/safe_two_stage_cleaned')
    p.add_argument('--model', choices=['Informer', 'iTransformer', 'PatchTST'],
                   help='Omit to use the existing SAFE device route')
    p.add_argument('--stage', choices=['both', 'regression', 'head'], default='both')
    p.add_argument('--backbone-checkpoint', type=Path,
                   help='Required for head-only mode: stage1_regression/best_backbone.pt')
    p.add_argument('--regression-loss', choices=['mse', 'mae'], default='mse')
    p.add_argument('--regression-epochs', type=int, default=100)
    p.add_argument('--head-learning-rate', type=float, default=1e-3)
    p.add_argument('--head-type', choices=['mlp', 'history'], default='mlp')
    p.add_argument('--head-loss', choices=['weighted_bce', 'asymmetric'], default='weighted_bce')
    p.add_argument('--asl-gamma-neg', type=float, default=4.)
    p.add_argument('--asl-gamma-pos', type=float, default=0.)
    p.add_argument('--asl-clip', type=float, default=.05)
    p.add_argument('--head-selection', choices=['loss', 'ap'], default='loss')
    p.add_argument('--threshold-policy', choices=['fixed', 'val_fbeta'], default='fixed')
    p.add_argument('--threshold-beta', type=float, default=1., help='1=F1; 2 emphasizes recall; validation only')
    return p


def state_hash(module):
    """Includes parameters AND buffers, notably BatchNorm running statistics."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


COMPATIBILITY_KEYS = ('device_id', 'model', 'seq_len', 'd_model', 'n_heads', 'd_ff',
                      'e_layers', 'd_layers', 'factor', 'dropout', 'patch_len', 'stride',
                      'feature_order', 'input_sha256', 'data_protocol')


def load_backbone(model, checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get('stage') != 'regression':
        raise ValueError('Expected a stage1 regression checkpoint, not a binary checkpoint')
    saved = checkpoint['config']
    differences = [key for key in COMPATIBILITY_KEYS if saved.get(key) != config.get(key)]
    if differences:
        raise ValueError('Incompatible backbone checkpoint fields: ' + ', '.join(differences))
    model.backbone.load_state_dict(checkpoint['backbone'], strict=True)
    if state_hash(model.backbone) != checkpoint['backbone_sha256']:
        raise ValueError('Backbone checkpoint state hash mismatch')
    return checkpoint


def run_epoch(model, loader, stage, criterion, device, optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0., 0
    squared = np.zeros(4, dtype=np.float64)
    absolute = np.zeros(4, dtype=np.float64)
    predictions, truths, indices = [], [], []
    with torch.set_grad_enabled(training):
        for x, xm, dm, binary_y, t, counts in loader:
            x, xm, dm = x.to(device), xm.to(device), dm.to(device)
            target = counts.to(device) if stage == 'regression' else binary_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model.forecast(x, xm, dm) if stage == 'regression' else model(x, xm, dm)
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite {stage} loss')
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()*len(target)
            count += len(target)
            if stage == 'regression':
                delta = (output.detach().double()-target.double()).cpu().numpy()
                squared += np.square(delta).sum(axis=0)
                absolute += np.abs(delta).sum(axis=0)
            if collect:
                predictions.append((output if stage == 'regression' else output.sigmoid()).detach().cpu().numpy())
                truths.append(target.detach().cpu().numpy())
                indices.extend(t.tolist())
    stats = {'loss': loss_sum/count, 'windows': count}
    if stage == 'regression':
        stats.update({'mse': float(squared.sum()/(count*4)), 'mae': float(absolute.sum()/(count*4)),
                      'per_level': {name: {'mse': float(squared[i]/count), 'mae': float(absolute[i]/count)}
                                    for i, name in enumerate(FEATURES)}})
    if collect:
        return stats, np.concatenate(truths), np.concatenate(predictions), np.asarray(indices)
    return stats


def fit_stage(model, loaders, stage, config, directory, device, criterion, epochs, lr):
    directory.mkdir(parents=True, exist_ok=False)
    params = model.backbone.parameters() if stage == 'regression' else model.head.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    frozen_hash = state_hash(model.backbone) if stage == 'head' else None
    if stage == 'head' and (not model.backbone_frozen or any(p.requires_grad for p in model.backbone.parameters())):
        raise RuntimeError('The backbone must be frozen before head training')
    best, stale, history = float('inf'), 0, []
    checkpoint_path = directory/('best_backbone.pt' if stage == 'regression' else 'best_head.pt')
    for epoch in range(1, epochs+1):
        train = run_epoch(model, loaders['train'], stage, criterion, device, optimizer)
        use_ap = stage == 'head' and config.get('head_selection') == 'ap'
        if use_ap:
            val, val_y, val_score, _ = run_epoch(model, loaders['val'], stage, criterion, device, collect=True)
            val['average_precision'] = metrics(val_y[:, 0].astype(int), (val_score[:, 0] >= .5).astype(int), val_score[:, 0])['average_precision']
        else:
            val = run_epoch(model, loaders['val'], stage, criterion, device)
        # Single-class validation has no meaningful ranking selection: fall back to loss.
        use_ap = use_ap and 0 < loaders['val'].dataset.summary['positive'] < loaders['val'].dataset.summary['windows']
        selection_value = -val['average_precision'] if use_ap else val['loss']
        if frozen_hash is not None and state_hash(model.backbone) != frozen_hash:
            raise RuntimeError('Backbone parameters or buffers changed during head training')
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
               'train_loss': train['loss'], 'val_loss': val['loss']}
        if stage == 'regression':
            row.update({f'{split}_{name}': stats[name] for split, stats in [('train', train), ('val', val)]
                        for name in ['mse', 'mae']})
        row['selection_metric'] = 'average_precision' if use_ap else 'loss'
        if 'average_precision' in val:
            row['val_average_precision'] = val['average_precision']
        history.append(row)
        pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
        print(stage, row, flush=True)
        if selection_value < best:
            best, stale = selection_value, 0
            checkpoint = {'stage': stage, 'config': config, 'epoch': epoch, 'validation': val,
                          'backbone_sha256': state_hash(model.backbone)}
            checkpoint['backbone' if stage == 'regression' else 'model'] = (
                model.backbone.state_dict() if stage == 'regression' else model.state_dict())
            torch.save(checkpoint, checkpoint_path)
        else:
            stale += 1
        if stale >= config['patience']:
            break
        for group in optimizer.param_groups:
            group['lr'] = lr * .5**(epoch-1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if stage == 'regression':
        model.backbone.load_state_dict(checkpoint['backbone'])
    else:
        model.load_state_dict(checkpoint['model'])
        if state_hash(model.backbone) != frozen_hash:
            raise RuntimeError('Reloaded head checkpoint changed the frozen backbone')
        write_json(directory/'freeze_audit.json', {'before_sha256': frozen_hash,
                   'after_sha256': state_hash(model.backbone), 'parameters_and_buffers_unchanged': True,
                   'optimizer_scope': 'head only', 'backbone_mode': 'eval'})
    return checkpoint_path, checkpoint


def main():
    args = configure(parser().parse_args())
    AsymmetricBinaryLoss(args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip)
    if not math.isfinite(args.threshold_beta) or args.threshold_beta <= 0:
        raise ValueError('threshold-beta must be finite and positive')
    if args.regression_epochs < 1 or args.head_learning_rate <= 0:
        raise ValueError('Invalid regression epochs or head learning rate')
    if (args.stage == 'head') != (args.backbone_checkpoint is not None):
        raise ValueError('Use --backbone-checkpoint exactly when --stage head')
    if args.backbone_checkpoint is not None and not args.backbone_checkpoint.is_file():
        raise FileNotFoundError(args.backbone_checkpoint)
    path = args.data_root/f'设备{args.device_id}_cleaned.csv'
    datasets = {s: TwoStageDataset(path, args.seq_len, s) for s in ['train', 'val', 'test']}
    audit = {s: d.summary for s, d in datasets.items()}
    print(json.dumps({'device': args.device_id, 'model': args.model, 'stage': args.stage,
                      'seq_len': args.seq_len, 'data_audit': audit}, ensure_ascii=False), flush=True)
    if not args.train:
        print('Preflight only: no model construction, no training, no output directory created.')
        return
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly if intended')
    positive, negative = audit['train']['positive'], audit['train']['negative']
    if args.stage != 'regression' and (not positive or not negative):
        raise ValueError('Head training requires both classes in the training windows')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SafeBinaryModel(args).to(device)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == 'train'),
                            num_workers=args.num_workers, drop_last=False) for s, d in datasets.items()}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'feature_order': FEATURES, 'label_rule': 'any_level_next_hour',
                   'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'data_audit': audit, 'data_protocol': 'raw_counts_gap_safe_70_10_20_v1',
                   'head_parameters': sum(p.numel() for p in model.head.parameters()),
                   'pos_weight': negative/positive if positive and args.head_loss == 'weighted_bce' else None,
                   'routing_scope': 'explicit model override or fixed full-cleaned-four-device route',
                   'torch_version': str(torch.__version__), 'numpy_version': np.__version__})
    if args.stage == 'head':
        load_backbone(model, args.backbone_checkpoint, config, device)
    run_dir = args.output_root/f'device{args.device_id}_{args.model}_L{args.seq_len}'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir/'config.json', config)
    print(f'Output: {run_dir}', flush=True)
    summary = {'device_id': args.device_id, 'model': args.model, 'seq_len': args.seq_len}
    backbone_path = args.backbone_checkpoint
    if args.stage != 'head':
        regression_criterion = nn.MSELoss() if args.regression_loss == 'mse' else nn.L1Loss()
        directory = run_dir/'stage1_regression'
        backbone_path, checkpoint = fit_stage(model, loaders, 'regression', config, directory,
                                              device, regression_criterion, args.regression_epochs, args.learning_rate)
        # Best epoch selected on validation, never on test metrics.
        report, true, pred, indices = run_epoch(model, loaders['test'], 'regression', regression_criterion, device, collect=True)
        report.update({'best_epoch': checkpoint['epoch'], 'optimized_loss': args.regression_loss})
        write_json(directory/'metrics.json', report)
        frame = {'target_time': datasets['test'].dates[indices].astype(str)}
        for i, name in enumerate(FEATURES):
            frame[f'{name}_true'], frame[f'{name}_pred'] = true[:, i], pred[:, i]
        pd.DataFrame(frame).to_csv(directory/'predictions.csv', index=False)
        summary['regression'] = report
        write_json(run_dir/'summary.json', summary)
    if args.stage != 'regression':
        # Explicit disk reload ensures the head uses the best, not the last backbone.
        source = load_backbone(model, backbone_path, config, device)
        config['backbone_source'] = str(backbone_path.resolve())
        config['backbone_best_epoch'] = source['epoch']
        config['backbone_sha256'] = source['backbone_sha256']
        write_json(run_dir/'config.json', config)
        model.freeze_backbone()
        if args.head_type == 'history':
            model.head.initialize_prior(positive, negative)
        criterion = (AsymmetricBinaryLoss(args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip)
                     if args.head_loss == 'asymmetric' else
                     nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative/positive], device=device)))
        directory = run_dir/'stage2_head'
        _, checkpoint = fit_stage(model, loaders, 'head', config, directory, device,
                                  criterion, args.epochs, args.head_learning_rate)
        if args.threshold_policy == 'val_fbeta':
            _, val_y, val_scores, val_indices = run_epoch(model, loaders['val'], 'head', criterion, device, collect=True)
            val_scores = val_scores.astype(np.float64)  # retain float32 values exactly in CSV
            pd.DataFrame({'target_time': datasets['val'].dates[val_indices].astype(str),
                          'y_true': val_y[:, 0], 'score': val_scores[:, 0]}).to_csv(directory/'validation_scores.csv', index=False)
            decision = select_threshold(val_y, val_scores, args.threshold_policy, args.threshold, args.threshold_beta)
        else:
            decision = select_threshold([0, 1], [.5, .5], 'fixed', args.threshold, args.threshold_beta)
        write_json(directory/'decision.json', decision)
        # Persist the operating point with the model before opening test predictions.
        checkpoint['decision'] = decision
        torch.save(checkpoint, directory/'best_head.pt')
        threshold = decision['threshold']
        stats, true, probability, indices = run_epoch(model, loaders['test'], 'head', criterion, device, collect=True)
        y, probability = true[:, 0].astype(int), probability[:, 0].astype(np.float64)
        predicted = (probability >= threshold).astype(int)
        report = {'best_epoch': checkpoint['epoch'], 'test_loss': stats['loss'], 'head_loss': args.head_loss,
                  'decision': decision, 'score_note': 'Sigmoid score is not a calibrated event probability',
                  'threshold': threshold, 'model': metrics(y, predicted, probability),
                  'always_normal': metrics(y, np.zeros_like(y)),
                  'persistence': metrics(y, datasets['test'].labels[indices-1].astype(int))}
        if args.head_loss == 'weighted_bce':
            report['test_bce'] = stats['loss']  # legacy consumers
        write_json(directory/'metrics.json', report)
        pd.DataFrame({'device_id': args.device_id, 'target_time': datasets['test'].dates[indices].astype(str),
                      'y_true': y, 'probability': probability, 'y_pred': predicted,
                      'threshold': threshold, 'model': args.model, 'seq_len': args.seq_len,
                      'label_rule': 'any_level_next_hour'}).to_csv(directory/'predictions.csv', index=False)
        summary['classification'] = report
    write_json(run_dir/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
