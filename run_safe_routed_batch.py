"""Plan/run/resume a complete routed frozen-backbone classification batch."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {'Informer', 'iTransformer', 'PatchTST'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def build_plan(routes_path, checkpoint_root, data_root, lengths, regression_loss):
    with Path(routes_path).open(encoding='utf-8-sig', newline='') as stream:
        routes = list(csv.DictReader(stream))
    if not routes:
        raise ValueError('Empty routing table')
    jobs, problems, seen = [], [], set()
    candidates = sorted(Path(checkpoint_root).glob('*/*/stage1_regression/best_backbone.pt'))
    for route in routes:
        device_name = route['Device_Name']
        # str.removeprefix requires Python 3.9; keep older servers compatible.
        device = device_name[len('设备'):] if device_name.startswith('设备') else device_name
        model = route['Model']
        if not device.isdigit() or device in seen or model not in MODELS:
            raise ValueError(f'Invalid or duplicate route: {device}/{model}')
        seen.add(device)
        data = Path(data_root)/f'设备{device}_cleaned.csv'
        if not data.is_file():
            problems.append(f'设备{device}: missing cleaned CSV {data}'); continue
        data_sha = digest(data)
        if route.get('Input_SHA256') and route['Input_SHA256'] != data_sha:
            problems.append(f'设备{device}: cleaned CSV differs from routing data hash'); continue
        for length in lengths:
            matched = []
            for path in candidates:
                # Filter by run config before loading potentially large model weights.
                cfg_path = path.parent.parent/'config.json'
                if not cfg_path.is_file():
                    continue
                cfg = json.loads(cfg_path.read_text())
                if (str(cfg.get('device_id')), cfg.get('model'), cfg.get('seq_len'), cfg.get('regression_loss')) != (device, model, length, regression_loss):
                    continue
                checkpoint = read_checkpoint(path)
                c = checkpoint.get('config', {})
                if checkpoint.get('stage') != 'regression' or 'backbone' not in checkpoint:
                    raise ValueError(f'Invalid regression checkpoint: {path}')
                if (str(c.get('device_id')), c.get('model'), c.get('seq_len'), c.get('regression_loss')) != (device, model, length, regression_loss):
                    raise ValueError(f'Run/checkpoint metadata mismatch: {path}')
                if c.get('input_sha256') != data_sha:
                    continue
                if c.get('data_protocol') != 'raw_counts_gap_safe_70_10_20_v1' or c.get('feature_order') != ['w_level1','w_level2','w_level3','w_level4']:
                    raise ValueError(f'Incompatible data protocol: {path}')
                value = float(checkpoint['validation'][regression_loss])
                if not (0 <= value < float('inf')):
                    raise ValueError(f'Invalid validation score: {path}')
                matched.append((value, str(path.resolve()), c))
            if not matched:
                problems.append(f'设备{device}/{model}/L{length}: no compatible {regression_loss} regression checkpoint'); continue
            score, selected, config = min(matched, key=lambda x: (x[0], x[1]))
            jobs.append({'id': f'device{device}_{model}_L{length}', 'device_id': device,
                         'model': model, 'seq_len': length, 'label': route.get('Label'),
                         'checkpoint': selected, 'checkpoint_sha256': digest(selected),
                         'data_path': str(data.resolve()), 'data_sha256': data_sha,
                         'validation_score': score, 'candidate_count': len(matched),
                         'config': config})
    if problems:
        raise ValueError('Batch incomplete; no training started:\n'+'\n'.join(problems))
    return {'routes': str(Path(routes_path).resolve()), 'routes_sha256': digest(routes_path),
            'devices': sorted(seen, key=int), 'lengths': lengths, 'regression_loss': regression_loss,
            'selection': 'lowest regression validation score; lexical path breaks ties; never test metrics',
            'jobs': jobs}


def command(job, output_root, args):
    c = job['config']
    cmd = [sys.executable, '-u', str(ROOT/'run_safe_two_stage.py'),
           '--device-id', job['device_id'], '--model', job['model'], '--seq-len', str(job['seq_len']),
           '--stage', 'head', '--backbone-checkpoint', job['checkpoint'],
           '--data-root', str(Path(job['data_path']).parent), '--output-root', str(output_root)]
    options = {'d-model': c['d_model'], 'n-heads': c['n_heads'], 'd-ff': c['d_ff'],
               'seed': c['seed'], 'regression-loss': c['regression_loss'],
               'head-hidden': c.get('head_hidden',16), 'head-dropout': c.get('head_dropout',.1),
               'batch-size': args.batch_size or c['batch_size'], 'epochs': args.epochs,
               'patience': args.patience, 'head-learning-rate': args.head_learning_rate,
               'threshold': args.threshold, 'device': args.device}
    for key, value in options.items():
        cmd.extend(['--'+key, str(value)])
    return cmd


def summarize(batch):
    import numpy as np
    import pandas as pd
    from run_safe_binary import metrics
    batch = Path(batch)
    plan = json.loads((batch/'plan.json').read_text())
    state = json.loads((batch/'state.json').read_text()) if (batch/'state.json').exists() else {}
    rows, missing = [], []
    for job in plan['jobs']:
        record = state.get(job['id'], {})
        if record.get('status') != 'complete':
            missing.append(job['id']); continue
        run = Path(record['run_dir'])
        df = pd.read_csv(run/'stage2_head/predictions.csv')
        if df.empty or df.isna().any().any():
            raise ValueError(f'Empty or missing predictions: {run}')
        for col, value in [('device_id',int(job['device_id'])),('model',job['model']),('seq_len',job['seq_len']),('label_rule','any_level_next_hour')]:
            if not df[col].eq(value).all():
                raise ValueError(f'Prediction metadata mismatch: {run}/{col}')
        if not df.y_true.isin([0,1]).all() or not df.y_pred.isin([0,1]).all() or not df.probability.between(0,1).all():
            raise ValueError(f'Invalid predictions: {run}')
        if not df.threshold.eq(plan['head_options']['threshold']).all() or not df.y_pred.eq((df.probability>=df.threshold).astype(int)).all():
            raise ValueError(f'Threshold mismatch: {run}')
        if not pd.to_datetime(df.target_time).is_unique:
            raise ValueError(f'Duplicate timestamps: {run}')
        report = metrics(df.y_true.to_numpy(), df.y_pred.to_numpy(), df.probability.to_numpy())
        row = {'device_id':job['device_id'],'model':job['model'],'seq_len':job['seq_len'],
               'windows':len(df),'threshold':float(df.threshold.iloc[0]), **report,
               'regression_validation_score':job['validation_score'],'backbone_checkpoint':job['checkpoint'],
               'run_dir':str(run)}
        rows.append(row)
    if rows:
        result = pd.DataFrame(rows).sort_values(['device_id','seq_len'])
        result.to_csv(batch/'all_metrics.csv',index=False,encoding='utf-8-sig')
        # Counts are aggregated separately per lookback: no repeated-window pooling across lookbacks.
        aggregations=[]
        for length in plan['lengths']:
            subset=result[result.seq_len==length]
            if len(subset)!=len(plan['devices']): continue
            tn,fp,fn,tp=[int(subset[k].sum()) for k in ['tn','fp','fn','tp']]
            aggregations.append({'seq_len':length,'devices':len(subset),
                'micro_accuracy':(tp+tn)/(tp+tn+fp+fn),'micro_precision':tp/(tp+fp) if tp+fp else 0.,
                'micro_recall':tp/(tp+fn) if tp+fn else None,
                'micro_f1':2*tp/(2*tp+fp+fn) if tp+fn else None,
                'micro_fpr':fp/(fp+tn) if fp+tn else None,
                **{'macro_'+k:float(subset[k].mean()) for k in ['accuracy','precision','recall','f1','average_precision','roc_auc']}})
        if aggregations: pd.DataFrame(aggregations).to_csv(batch/'by_length.csv',index=False,encoding='utf-8-sig')
        display=['device_id','model','seq_len','accuracy','precision','recall','f1','average_precision','roc_auc','false_positive_rate']
        lines=['# 按聚类路由的分类结果','',f"完成 {len(rows)}/{len(plan['jobs'])} 组。",'',
               '| '+' | '.join(display)+' |','| '+' | '.join(['---']*len(display))+' |']
        for _,r in result.iterrows():
            lines.append('| '+' | '.join(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k in display)+' |')
        lines+=['','缺失实验：'+(', '.join(missing) or '无'),'',
                '所有阈值指标由各实验predictions.csv重算；AP使用average precision。按长度汇总同时报告设备宏平均及混淆矩阵微平均，不跨历史长度重复汇总同一窗口。主干按回归验证指标选择，不按测试指标挑选。']
        (batch/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(batch/'completion.json',{'expected':len(plan['jobs']),'completed':len(rows),'missing':missing,'complete':not missing})
    return not missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,default=ROOT/'docs/safe_cleaned_four_devices_clustering/device_model_assignments.csv')
    p.add_argument('--checkpoint-root',type=Path,default=ROOT/'outputs/safe_two_stage_cleaned')
    p.add_argument('--data-root',type=Path,default=ROOT/'dataset/fault_selected_cleaned')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/safe_routed_classification')
    p.add_argument('--lengths',type=int,nargs='+',default=[3,6,12,24],choices=[3,6,12,24])
    p.add_argument('--regression-loss',choices=['mse','mae'],default='mse')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path,help='Resume exactly the saved plan and settings')
    p.add_argument('--summarize',type=Path,help='Only recompute summaries of an existing batch')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=0)
    p.add_argument('--head-learning-rate',type=float,default=.001)
    p.add_argument('--threshold',type=float,default=.5)
    a=p.parse_args()
    if a.summarize:
        if a.execute or a.resume: p.error('--summarize cannot be combined with execute/resume')
        if not summarize(a.summarize): raise SystemExit('Incomplete batch: see completion.json')
        return
    if a.epochs<1 or a.patience<1 or a.batch_size<0 or a.head_learning_rate<=0 or not 0<a.threshold<1:
        p.error('Invalid head settings')
    if len(a.lengths)!=len(set(a.lengths)): p.error('Duplicate lengths')
    options=['device','epochs','patience','batch_size','head_learning_rate','threshold']
    if a.resume:
        batch=a.resume.resolve();plan=json.loads((batch/'plan.json').read_text())
        for key,value in plan['head_options'].items(): setattr(a,key,value)
    else:
        plan=build_plan(a.routes,a.checkpoint_root,a.data_root,a.lengths,a.regression_loss)
        batch=a.output_root.resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        batch.mkdir(parents=True,exist_ok=False)
        plan['head_options']={key:getattr(a,key) for key in options}
        save(batch/'plan.json',plan)
    print(f"Batch: {batch}\nDevices: {plan['devices']}\nJobs: {len(plan['jobs'])}",flush=True)
    # Revalidate all pinned files before any training, including resume.
    for job in plan['jobs']:
        if digest(job['data_path'])!=job['data_sha256'] or digest(job['checkpoint'])!=job['checkpoint_sha256']:
            raise ValueError(f"Pinned input changed: {job['id']}")
    if not a.execute:
        print('Plan only; no training. Start with --resume THIS_BATCH --execute.');return
    # Advisory lock is released by the OS even when the process is interrupted.
    import fcntl
    batch_lock = (batch/'execute.lock').open('a')
    try:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This batch is already running in another process')
    state_path=batch/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for job in plan['jobs']:
        if state.get(job['id'],{}).get('status')=='complete': continue
        # Each attempt has its own output root; never pick an old unrelated run.
        attempt=batch/'jobs'/job['id']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        attempt.mkdir(parents=True,exist_ok=False)
        cmd=command(job,attempt,a)+['--train']
        state[job['id']]={'status':'running','attempt':str(attempt),'command':cmd};save(state_path,state)
        print(f"Running {job['id']}; log: {attempt/'train.log'}",flush=True)
        with (attempt/'train.log').open('w') as stream:
            completed=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        runs=list(attempt.glob('*/*/summary.json'))
        if completed.returncode or len(runs)!=1:
            state[job['id']]['status']='failed';save(state_path,state);summarize(batch)
            raise SystemExit(f"Failed {job['id']}; see {attempt/'train.log'}")
        run=runs[0].parent
        freeze=json.loads((run/'stage2_head/freeze_audit.json').read_text())
        if not freeze['parameters_and_buffers_unchanged'] or freeze['before_sha256']!=freeze['after_sha256']:
            raise RuntimeError(f"Frozen backbone check failed: {job['id']}")
        state[job['id']].update(status='complete',run_dir=str(run));save(state_path,state)
        summarize(batch)
    if not summarize(batch): raise SystemExit('Batch incomplete')
    print(f'Complete: {batch}/REPORT.md',flush=True)


if __name__=='__main__':
    main()
