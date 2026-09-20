"""Routed two-stage next-hour fault/type prediction; raw logs are required."""
import argparse
import csv
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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from data_provider.fault_type_data import prepare, FaultTypeDataset
from models.SAFE_FaultTypes import FaultTypeModel

ROOT = Path(__file__).resolve().parent
DEFAULT_ROUTES = ROOT/'docs/safe_all_selected_cleaned_clustering/device_model_assignments.csv'


class TwoStageTypeDataset(FaultTypeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Next-hour regression target is returned separately, never passed as history.
        target = torch.from_numpy(self.data['values'][sample[-1]].copy())
        return (*sample, target)


def build_plan(devices, lengths, seed, routes):
    with Path(routes).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        name = row['Device_Name']
        if name not in {f'设备{d}' for d in devices}:
            continue
        if name in selected or row['Model'] not in ('Informer', 'iTransformer', 'PatchTST'):
            raise ValueError('Duplicate or unsupported device route: '+name)
        selected[name] = row['Model']
    missing = [d for d in devices if f'设备{d}' not in selected]
    if missing:
        raise ValueError(f'Missing device routes: {missing}')
    return [dict(device=d, length=length, model=selected[f'设备{d}'], seed=seed)
            for d in devices for length in lengths]


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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


def epoch(model, loader, criterion, device, optimizer=None, stage='head'):
    model.train(optimizer is not None)
    total, ys, scores, ts = 0., [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, xm, dm, y, t, regression_target in loader:
            output_fn = model.forecast if stage == 'regression' else model
            logits = output_fn(x.to(device), xm.to(device), dm.to(device))
            target = regression_target if stage == 'regression' else y
            loss = criterion(logits, target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            total += loss.item()*len(y)
            ys.append(target.numpy())
            scores.append((logits if stage == 'regression' else logits.sigmoid()).detach().cpu().numpy())
            ts.extend(t.tolist())
    return total/len(loader.dataset), np.concatenate(ys), np.concatenate(scores), np.array(ts)



def fit_stage(model, loaders, criterion, device, directory, config, audit,
              stage, epochs, patience, learning_rate, weights=None):
    directory.mkdir()
    if stage == 'head' and (not model.backbone_frozen or any(p.requires_grad for p in model.backbone.parameters())):
        raise RuntimeError('Classification requires a frozen pretrained backbone')
    module = model.backbone if stage == 'regression' else model.head
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    frozen = state_hash(model.backbone) if stage == 'head' else None
    checkpoint = directory/('best_backbone.pt' if stage == 'regression' else 'best_head.pt')
    history, best, stale = [], float('inf'), 0
    for e in range(1, epochs+1):
        train_loss, _, _, _ = epoch(model, loaders['train'], criterion, device, optimizer, stage)
        val_loss, _, _, _ = epoch(model, loaders['val'], criterion, device, stage=stage)
        if frozen is not None and state_hash(model.backbone) != frozen:
            raise RuntimeError('Frozen backbone parameters or buffers changed')
        history.append(dict(epoch=e, train_loss=train_loss, val_loss=val_loss,
                            learning_rate=optimizer.param_groups[0]['lr']))
        pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
        print(directory.parent.name, stage, history[-1], flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            saved = dict(stage=stage, config=config, audit=audit, epoch=e, val_loss=val_loss,
                         backbone_sha256=state_hash(model.backbone))
            saved['backbone' if stage == 'regression' else 'model'] = (
                model.backbone.state_dict() if stage == 'regression' else model.state_dict())
            if weights is not None:
                saved['pos_weight'] = weights.tolist()
            torch.save(saved, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
        for group in optimizer.param_groups:
            group['lr'] = learning_rate * .5**(e-1)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if stage == 'regression':
        model.backbone.load_state_dict(saved['backbone'], strict=True)
    else:
        model.load_state_dict(saved['model'], strict=True)
        if state_hash(model.backbone) != frozen:
            raise RuntimeError('Checkpoint reload changed frozen backbone')
        write_json(directory/'freeze_audit.json', dict(before_sha256=frozen,
            after_sha256=state_hash(model.backbone), parameters_and_buffers_unchanged=True,
            optimizer_scope='head only', backbone_mode='eval'))
    return saved

def run(args, device_id, length, model_name, seed, batch, data):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    datasets = {s: TwoStageTypeDataset(data, length, s) for s in ('train', 'val', 'test')}
    directory = batch/f'device{device_id}_{model_name}_L{length}_seed{seed}'
    directory.mkdir()
    audit = data['audit']
    write_json(directory/'data_audit.json', dict(audit, splits={s: d.summary for s, d in datasets.items()}))
    names, catalog = audit['names'], audit['catalog']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device_id=device_id, seq_len=length, model=model_name, seed=seed,
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  protocol='fault_types_two_stage_v1',
                  routes_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                  regression_target='next-hour standardized log1p severity and training-type counts')
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
    regression_dir = directory/'stage1_regression'
    regression = fit_stage(model, loaders, nn.MSELoss(), device, regression_dir,
                           config, audit, 'regression', args.regression_epochs,
                           args.patience, args.learning_rate)
    # The best validation-MSE checkpoint has been loaded before freezing.
    model.freeze_backbone()
    config['backbone_checkpoint'] = str(regression_dir/'best_backbone.pt')
    config['backbone_sha256'] = regression['backbone_sha256']
    config['regression_best_epoch'] = regression['epoch']
    write_json(directory/'config.json', config)
    directory = directory/'stage2_head'
    saved = fit_stage(model, loaders, criterion, device, directory, config, audit,
                      'head', args.epochs, args.patience, args.head_learning_rate, weights)
    _, vy, vs, _ = epoch(model, loaders['val'], criterion, device)
    cuts = thresholds(vy, vs)
    saved['thresholds'] = cuts.tolist(); saved['supported_types'] = supported.tolist()
    torch.save(saved, directory/'best_head.pt')
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
    regression_loss, truth, prediction, _ = epoch(model, loaders['test'], nn.MSELoss(), device, stage='regression')
    write_json(regression_dir/'metrics.json', dict(mse=regression_loss,
        mae=float(np.abs(truth-prediction).mean()), space='standardized log1p counts',
        best_epoch=regression['epoch']))
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
                best_epoch=saved['epoch'], regression_best_epoch=regression['epoch'], path=str(directory))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=ROOT/'dataset/fault_raw/raw_data')
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs/safe_fault_types')
    p.add_argument('--devices', nargs='+', type=int, default=[27,28,39,46,58])
    p.add_argument('--lengths', nargs='+', type=int, choices=[6,12], default=[6,12])
    p.add_argument('--routes', type=Path, default=DEFAULT_ROUTES)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--regression-epochs', type=int, default=100)
    p.add_argument('--head-learning-rate', type=float, default=.001)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.0001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train', action='store_true')
    args = p.parse_args()
    if min(args.lengths+args.devices+[args.regression_epochs,args.epochs,args.patience,args.batch_size,args.threads,args.d_model]) < 1 or args.d_model % 4 or args.learning_rate <= 0 or args.head_learning_rate <= 0:
        p.error('Invalid positive training parameters; d-model must be divisible by 4')
    for key in ('devices','lengths'):
        if len(getattr(args,key)) != len(set(getattr(args,key))):
            p.error('Duplicate '+key)
    if args.train and tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) < (2, 1):
        p.error('Training requires PyTorch >=2.1.0 for safe checkpoint loading')
    if args.train and args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu explicitly')
    if not 0 <= args.seed < 2**32:
        p.error('seed must lie in [0, 2**32)')
    plan = build_plan(args.devices, args.lengths, args.seed, args.routes)
    torch.set_num_threads(args.threads)
    batch = args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch.mkdir(parents=True)
    expected = len(plan)
    write_json(batch/'plan.json', dict(protocol='fault_types_two_stage_v1', jobs=plan,
        routes=str(args.routes), routes_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest()))
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
    for job in plan:
        d, length, model, seed = (job[k] for k in ('device', 'length', 'model', 'seed'))
        try:
            rows.append(run(args, d, length, model, seed, batch, prepared[d]))
        except Exception as exc:
            write_json(batch/'completion.json', dict(complete=False, expected=expected,
                completed=len(rows), training=args.train, error=str(exc), failed_job=job))
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
