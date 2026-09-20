"""Next-hour joint fault occurrence/type experiments; raw logs are required."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from data_provider.fault_type_data import prepare, FaultTypeDataset
from models.SAFE_FaultTypes import FaultTypeModel

ROOT = Path(__file__).resolve().parent


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def thresholds(y, scores):
    result = np.full(y.shape[1], .5)
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            grid = np.linspace(.05, .95, 19)
            # Highest threshold wins ties, reducing false alerts.
            result[j] = max(grid, key=lambda t: (f1_score(y[:, j], scores[:, j] >= t, zero_division=0), t))
    return result


def decode(scores, cuts, supported):
    binary = scores[:, 0] >= cuts[0]
    types = (scores[:, 1:] >= cuts[1:]) & supported[None, :]
    # Binary gate makes no-fault output an empty set. An untyped alert is explicit.
    types &= binary[:, None]
    unknown = binary & ~types.any(1)
    return binary, types, unknown


def binary_metrics(y, pred, score):
    y, pred = y.astype(bool), pred.astype(bool)
    return dict(precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)), support=int(y.sum()),
                tn=int((~y & ~pred).sum()), fp=int((~y & pred).sum()),
                fn=int((y & ~pred).sum()), tp=int((y & pred).sum()),
                average_precision=float(average_precision_score(y, score)) if y.any() else None)


def report(y, pred, score, names, binary_y, binary_pred, binary_score):
    per_type = {name: binary_metrics(y[:, j], pred[:, j], score[:, j]) for j, name in enumerate(names)}
    supported = y.sum(0) > 0
    return dict(binary=binary_metrics(binary_y, binary_pred, binary_score),
                type_micro_f1=float(f1_score(y.ravel(), pred.ravel(), zero_division=0)),
                type_macro_f1_present=float(np.mean([v['f1'] for v in per_type.values() if v['support']])) if supported.any() else None,
                type_macro_f1_all=float(np.mean([v['f1'] for v in per_type.values()])),
                exact_type_set_accuracy=float((y == pred).all(1).mean()), per_type=per_type)


def epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total, ys, scores, ts = 0., [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, xm, dm, y, t in loader:
            logits = model(x.to(device), xm.to(device), dm.to(device))
            loss = criterion(logits, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            total += loss.item()*len(y)
            ys.append(y.numpy()); scores.append(logits.detach().sigmoid().cpu().numpy()); ts.extend(t.tolist())
    return total/len(loader.dataset), np.concatenate(ys), np.concatenate(scores), np.array(ts)


def run(args, device_id, length, model_name, seed, batch, data):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    datasets = {s: FaultTypeDataset(data, length, s) for s in ('train', 'val', 'test')}
    directory = batch/f'device{device_id}_{model_name}_L{length}_seed{seed}'
    directory.mkdir()
    audit = data['audit']
    write_json(directory/'data_audit.json', dict(audit, splits={s: d.summary for s, d in datasets.items()}))
    names, catalog = audit['names'], audit['catalog']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device_id=device_id, seq_len=length, model=model_name, seed=seed,
                  torch_version=str(torch.__version__), numpy_version=np.__version__)
    write_json(directory/'config.json', config)
    if not args.train:
        return dict(device=device_id, model=model_name, length=length, seed=seed, status='preflight')
    device = torch.device(args.device)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s=='train') for s, d in datasets.items()}
    train_y = np.c_[data['binary'][datasets['train'].targets], data['labels'][datasets['train'].targets]]
    positives = train_y.sum(0)
    weights = np.clip((len(train_y)-positives)/np.maximum(positives, 1), 1, 50)
    supported = positives[1:] > 0
    model = FaultTypeModel(model_name, length, data['values'].shape[1], len(names), args.d_model).to(device)
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
    # Equal task contribution independent of number of fault types.
    def criterion(logits, y):
        losses = bce(logits, y)
        return losses[:, 0].mean()+losses[:, 1:].mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, best, stale = [], float('inf'), 0
    for e in range(1, args.epochs+1):
        train_loss, _, _, _ = epoch(model, loaders['train'], criterion, device, optimizer)
        val_loss, _, _, _ = epoch(model, loaders['val'], criterion, device)
        history.append(dict(epoch=e, train_loss=train_loss, val_loss=val_loss))
        pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
        print(directory.name, history[-1], flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(model=model.state_dict(), config=config, audit=audit,
                            pos_weight=weights.tolist(), epoch=e, val_loss=val_loss), directory/'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
    saved = torch.load(directory/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['model'])
    _, vy, vs, _ = epoch(model, loaders['val'], criterion, device)
    cuts = thresholds(vy, vs)
    saved['thresholds'] = cuts.tolist(); saved['supported_types'] = supported.tolist()
    torch.save(saved, directory/'best.pt')
    write_json(directory/'thresholds.json', dict(names=['any_fault']+names, values=cuts.tolist(),
                selection='validation F1 per output; .5 for single-class validation', supported=supported.tolist()))
    _, _, scores, targets = epoch(model, loaders['test'], criterion, device)
    scores[:, 1:][:, ~supported] = 0.
    binary, pred, unknown = decode(scores, cuts, supported)
    full_y = data['counts'][targets] > 0
    full_pred = np.zeros_like(full_y)
    full_score = np.zeros(full_y.shape, dtype=np.float32)
    cols = [catalog.index(name) for name in names]
    full_pred[:, cols], full_score[:, cols] = pred, scores[:, 1:]
    by = data['binary'][targets]
    result = report(full_y, full_pred, full_score, catalog, by, binary, scores[:, 0])
    # Unknown is an additional emitted label, so it cannot count as an exact typed match.
    result['exact_type_set_accuracy'] = float(((full_y == full_pred).all(1) & ~unknown).mean())
    previous = data['counts'][targets-1] > 0
    result['persistence_baseline'] = report(full_y, previous, previous.astype(float), catalog,
                                            by, previous.any(1), previous.any(1).astype(float))
    result['all_normal_baseline'] = report(full_y, np.zeros_like(full_y), np.zeros(full_y.shape), catalog,
                                           by, np.zeros(len(by), dtype=bool), np.zeros(len(by)))
    result.update(untyped_alerts=int(unknown.sum()), unseen_test_events=int(data['counts'][targets][:, [i for i,n in enumerate(catalog) if n not in names]].sum()),
                  best_epoch=saved['epoch'], test_windows=len(targets), thresholds_source='validation only')
    write_json(directory/'metrics.json', result)
    rows = []
    for i, t in enumerate(targets):
        predicted = [name for j, name in enumerate(names) if pred[i, j]]
        if unknown[i]:
            predicted = ['__UNKNOWN_TYPE__']
        rows.append(dict(target_hour=str(data['dates'][t]), any_fault_true=int(by[i]),
                         any_fault_pred=int(binary[i]), any_fault_score=float(scores[i, 0]),
                         true_types=json.dumps([n for j,n in enumerate(catalog) if full_y[i,j]], ensure_ascii=False),
                         predicted_types=json.dumps(predicted, ensure_ascii=False),
                         type_scores=json.dumps(dict(zip(names, map(float, scores[i,1:]))), ensure_ascii=False)))
    pd.DataFrame(rows).to_csv(directory/'predictions.csv', index=False)
    return dict(device=device_id, model=model_name, length=length, seed=seed, status='complete',
                binary_f1=result['binary']['f1'], type_micro_f1=result['type_micro_f1'],
                type_macro_f1_present=result['type_macro_f1_present'],
                persistence_type_micro_f1=result['persistence_baseline']['type_micro_f1'],
                best_epoch=saved['epoch'], path=str(directory))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=ROOT/'dataset/fault_raw/raw_data')
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs/safe_fault_types')
    p.add_argument('--devices', nargs='+', type=int, default=[27,28,39,46,58])
    p.add_argument('--lengths', nargs='+', type=int, default=[3,6,12,24])
    p.add_argument('--models', nargs='+', choices=['Informer','iTransformer','PatchTST'], default=['iTransformer'])
    p.add_argument('--seeds', nargs='+', type=int, default=[2021,2022,2023])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train', action='store_true')
    args = p.parse_args()
    if min(args.lengths+args.devices+[args.epochs,args.patience,args.batch_size,args.threads,args.d_model]) < 1 or args.d_model % 4 or args.learning_rate <= 0:
        p.error('Invalid positive training parameters; d-model must be divisible by 4')
    for key in ('devices','lengths','models','seeds'):
        if len(getattr(args,key)) != len(set(getattr(args,key))):
            p.error('Duplicate '+key)
    if args.train and tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) < (2, 2):
        p.error('Training requires PyTorch >=2.2 for safe checkpoint loading')
    if args.train and args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu explicitly')
    torch.set_num_threads(args.threads)
    batch = args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch.mkdir(parents=True)
    expected = len(args.devices)*len(args.lengths)*len(args.models)*len(args.seeds)
    write_json(batch/'completion.json', dict(complete=False, expected=expected, completed=0, training=args.train))
    print('Batch:', batch, flush=True)
    # Validate every input/window before any model updates; reuse immutable prepared arrays.
    prepared = {d: prepare(args.raw_root/f'设备{d}_NetworkFaultLog.csv') for d in args.devices}
    for d, data in prepared.items():
        for length in args.lengths:
            for split in ('train', 'val', 'test'):
                FaultTypeDataset(data, length, split)
        target = batch/'prepared'/f'设备{d}'
        target.mkdir(parents=True)
        frame = pd.DataFrame(data['counts'].astype(int), columns=['type::'+n for n in data['audit']['catalog']])
        frame.insert(0, 'date', data['dates'])
        for j in range(4):
            frame[f'w_level{j+1}'] = data['severity'][:, j].astype(int)
        frame.to_csv(target/'hourly.csv', index=False)
        write_json(target/'audit.json', data['audit'])
    rows = []
    for d in args.devices:
        for length in args.lengths:
            for model in args.models:
                for seed in args.seeds:
                    try:
                        rows.append(run(args, d, length, model, seed, batch, prepared[d]))
                    except Exception as exc:
                        write_json(batch/'completion.json', dict(complete=False, expected=expected,
                            completed=len(rows), training=args.train, error=str(exc),
                            failed_job=dict(device=d, length=length, model=model, seed=seed)))
                        raise
                    pd.DataFrame(rows).to_csv(batch/'summary.csv', index=False)
                    write_json(batch/'completion.json', dict(complete=False, expected=expected,
                        completed=len(rows), training=args.train))
    if args.train:
        pd.DataFrame(rows).groupby(['device','model','length'])[
            ['binary_f1','type_micro_f1','type_macro_f1_present','persistence_type_micro_f1']
        ].agg(['mean','std']).to_csv(batch/'seed_summary.csv')
    write_json(batch/'completion.json', dict(complete=True, expected=expected, completed=len(rows), training=args.train))
    print('Complete:', batch, flush=True)


if __name__ == '__main__':
    main()
