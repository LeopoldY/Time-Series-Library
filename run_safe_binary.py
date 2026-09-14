"""Standalone SAFE binary adaptation. Default: inspect data only; --train opts in."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from data_provider.safe_binary_data import SafeBinaryDataset, FEATURES
from models.SAFE_Binary import SafeBinaryModel

ROOT = Path(__file__).resolve().parent
ROUTES = {'27': 'iTransformer', '58': 'Informer', '69': 'iTransformer', '83': 'PatchTST'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device-id', required=True, choices=ROUTES)
    p.add_argument('--data-root', type=Path, default=ROOT/'dataset/fault_selected_cleaned')
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs/safe_binary_cleaned')
    p.add_argument('--seq-len', type=int, default=24, choices=[3, 6, 12, 24])
    p.add_argument('--train', action='store_true', help='Explicitly enable training')
    p.add_argument('--device', default='cuda:0', help='cpu or cuda:N')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=0, help='0: original routed default')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--head-hidden', type=int, default=16)
    p.add_argument('--head-dropout', type=float, default=.1)
    p.add_argument('--threshold', type=float, default=.5, help='Fixed before test evaluation')
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=2048)
    return p


def configure(args):
    args.model = ROUTES[args.device_id]
    args.e_layers = 3 if args.model == 'Informer' else 2
    args.d_layers, args.factor, args.dropout = 1, 3, .1
    args.patch_len = 1 if args.seq_len == 3 else args.seq_len // 2
    args.stride = max(1, args.patch_len // 2)
    args.batch_size = args.batch_size or (16 if args.model == 'PatchTST' else 32)
    if (args.head_hidden < 1 or args.epochs < 1 or args.patience < 1
            or args.batch_size < 1 or args.num_workers < 0 or args.learning_rate <= 0
            or not 0 < args.threshold < 1 or not 0 <= args.head_dropout < 1
            or args.d_model < 2 or args.n_heads < 1 or args.d_ff < 1
            or args.d_model % args.n_heads or args.d_model % 2):
        raise ValueError('Invalid training/model configuration')
    return args


def evaluate(model, loader, criterion, device):
    model.eval()
    total, count, labels, probabilities, indices = 0., 0, [], [], []
    with torch.no_grad():
        for x, xm, dm, y, t in loader:
            logits = model(x.to(device), xm.to(device), dm.to(device))
            loss = criterion(logits, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite evaluation loss')
            total += loss.item() * len(y)
            count += len(y)
            labels.extend(y[:, 0].tolist())
            probabilities.extend(logits.sigmoid()[:, 0].cpu().tolist())
            indices.extend(t.tolist())
    return total/count, np.asarray(labels, dtype=int), np.asarray(probabilities), indices


def metrics(y, prediction, probability=None):
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    positives, negatives = tp+fn, tn+fp
    result = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'positive': positives,
              'negative': negatives, 'accuracy': (tp+tn)/len(y),
              'precision': tp/(tp+fp) if tp+fp else 0.,
              'recall': tp/positives if positives else None,
              'f1': 2*tp/(2*tp+fp+fn) if positives else None,
              'false_positive_rate': fp/negatives if negatives else None}
    if probability is not None:
        result['average_precision'] = float(average_precision_score(y, probability)) if positives else None
        result['roc_auc'] = float(roc_auc_score(y, probability)) if positives and negatives else None
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    args = configure(parser().parse_args())
    path = args.data_root/f'设备{args.device_id}_cleaned.csv'
    datasets = {s: SafeBinaryDataset(path, args.seq_len, s) for s in ['train', 'val', 'test']}
    audit = {s: d.summary for s, d in datasets.items()}
    print(json.dumps({'device': args.device_id, 'model': args.model, 'seq_len': args.seq_len,
                      'data_audit': audit}, ensure_ascii=False, indent=2), flush=True)
    if not args.train:
        print('Preflight only: no model construction, no training, no output directory created.')
        return
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly if intended')
    positive, negative = audit['train']['positive'], audit['train']['negative']
    if not positive or not negative:
        raise ValueError('Training windows must contain both classes')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SafeBinaryModel(args).to(device)
    pos_weight = negative/positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s=='train'),
                            num_workers=args.num_workers, drop_last=False)
               for s, d in datasets.items()}
    run_dir = args.output_root/f'device{args.device_id}_{args.model}_L{args.seq_len}'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'label_rule': 'any(w_level1..4 at next hour > 0)', 'feature_order': FEATURES,
                   'head_parameters': sum(p.numel() for p in model.head.parameters()),
                   'pos_weight': pos_weight, 'data_audit': audit,
                   'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'routing_scope': 'fixed from full cleaned four-device clustering',
                   'torch_version': str(torch.__version__), 'numpy_version': np.__version__})
    write_json(run_dir/'config.json', config)
    print(f'Output: {run_dir}', flush=True)
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.; count = 0
        for x, xm, dm, y, _ in loaders['train']:
            optimizer.zero_grad()
            logits = model(x.to(device), xm.to(device), dm.to(device))
            loss = criterion(logits, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward(); optimizer.step()
            total += loss.item()*len(y); count += len(y)
        val_loss, _, _, _ = evaluate(model, loaders['val'], criterion, device)
        history.append({'epoch': epoch, 'train_bce': total/count, 'val_bce': val_loss,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(run_dir/'history.csv', index=False)
        print(history[-1], flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                        'val_bce': best}, run_dir/'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
        # Original type1 schedule: halve after the first completed epoch.
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * .5**(epoch-1)
    checkpoint = torch.load(run_dir/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test_loss, y, probability, indices = evaluate(model, loaders['test'], criterion, device)
    prediction = (probability >= args.threshold).astype(int)
    test = datasets['test']
    persistence = test.labels[np.asarray(indices)-1].astype(int)
    report = {'best_epoch': checkpoint['epoch'], 'test_bce': test_loss,
              'threshold': args.threshold, 'model': metrics(y, prediction, probability),
              'always_normal': metrics(y, np.zeros_like(y)),
              'persistence': metrics(y, persistence),
              'note': 'Weighted sigmoid scores are not calibrated probabilities; null metrics are undefined.'}
    write_json(run_dir/'metrics.json', report)
    pd.DataFrame({'device_id': args.device_id, 'target_time': test.dates[indices].astype(str),
                  'y_true': y, 'probability': probability, 'y_pred': prediction,
                  'threshold': args.threshold, 'model': args.model, 'seq_len': args.seq_len,
                  'label_rule': 'any_level_next_hour'}).to_csv(run_dir/'predictions.csv', index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
