import numpy as np
import pandas as pd
import pytest
import torch
from data_provider.fault_type_data import prepare, FaultTypeDataset
from models.SAFE_FaultTypes import FaultTypeModel
from run_safe_fault_types import decode, thresholds


def source(tmp_path):
    rows = [
        (1, ' A ', '提示', '设备27', '01/01/2013 00:00:00'),
        (2, 'B', '重要', '设备27', '01/01/2013 01:05:00'),
        (3, 'A', '提示', '设备27', '01/01/2013 01:10:00'),
        (4, 'NEW', '紧急', '设备27', '01/05/2013 12:00:00'),
        (5, 'END', '提示', '设备27', '01/06/2013 00:30:00'),
    ]
    df = pd.DataFrame(rows, columns=['ID','名称','级别','告警源','发生时间'])
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    path = tmp_path/'设备27_NetworkFaultLog.csv'
    df.to_csv(path, index=False)
    return path


def test_cleaning_causal_vocabulary_and_boundaries(tmp_path):
    path = source(tmp_path)
    d = prepare(path)
    assert d['audit']['names'] == ['A','B']
    assert d['audit']['unseen_types'] == ['NEW']
    assert d['audit']['duplicate_ids_removed'] == 1
    assert d['audit']['partial_boundary_events_removed'] == 1
    assert d['counts'].sum() == 4
    assert (d['counts'][1] > 0).sum() == 2  # simultaneous labels retained
    assert len(d['dates']) == 120  # quiet hours remain
    assert np.allclose(d['values'][:84].mean(0), 0, atol=1e-5)
    splits = {s: FaultTypeDataset(d, 3, s) for s in ('train','val','test')}
    assert splits['train'].targets.max() < splits['val'].targets.min()
    assert splits['val'].targets.max() < splits['test'].targets.min()
    x, _, _, y, t = splits['test'][0]
    np.testing.assert_array_equal(x.numpy(), d['values'][t-3:t])
    assert t not in range(t-3,t)
    # Changing only future names must not change input schema or train normalization.
    f = pd.read_csv(path); f.loc[f['名称']=='NEW','名称']='DIFFERENT'; f.to_csv(path,index=False)
    changed = prepare(path)
    assert changed['audit']['names'] == d['audit']['names']
    np.testing.assert_array_equal(changed['values'][:84], d['values'][:84])


def test_conflicting_ids_rejected(tmp_path):
    path = source(tmp_path)
    f = pd.read_csv(path); f.loc[len(f)-1,'名称']='conflict'; f.to_csv(path,index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        prepare(path)


def test_binary_type_consistency_and_unknown():
    score = np.array([[.1,.9,.9],[.9,.1,.9],[.9,.9,.1]])
    binary, types, unknown = decode(score, np.array([.5,.5,.5]), np.array([True,False]))
    assert binary.tolist() == [False,True,True]
    assert types.tolist() == [[False,False],[False,False],[True,False]]
    assert unknown.tolist() == [False,True,False]
    assert np.array_equal(binary, types.any(1)|unknown)
    assert thresholds(np.zeros((3,2)), score[:,:2]).tolist() == [.5,.5]


@pytest.mark.parametrize('model', ['iTransformer','Informer','PatchTST'])
@pytest.mark.parametrize('length', [3,24])
def test_backbones_joint_gradient_and_reload(model, length, tmp_path):
    torch.set_num_threads(1)
    net = FaultTypeModel(model,length,6,2,16)
    x, xm, dm = torch.randn(4,length,6), torch.randn(4,length,4), torch.randn(4,2,4)
    y = net(x,xm,dm)
    assert y.shape == (4,3)
    torch.nn.functional.binary_cross_entropy_with_logits(y,torch.ones_like(y)).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.backbone.parameters())
    assert torch.isfinite(y).all()
    path = tmp_path/'checkpoint.pt'; torch.save(net.state_dict(),path)
    other = FaultTypeModel(model,length,6,2,16)
    other.load_state_dict(torch.load(path,weights_only=True))
    for a,b in zip(net.parameters(),other.parameters()):
        assert torch.equal(a,b)


def test_routed_preflight_artifacts(tmp_path, monkeypatch):
    import json
    import sys
    import run_safe_fault_types as runner
    source(tmp_path)
    out = tmp_path/'results'
    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must not construct a model or train')
    monkeypatch.setattr(runner, 'FaultTypeModel', forbidden)
    monkeypatch.setattr(runner, 'fit_stage', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_safe_fault_types.py', '--raw-root', str(tmp_path),
        '--output-root', str(out), '--devices', '27', '--lengths', '6', '12',
        '--seed', '42', '--threads', '1', '--device', 'cpu'])
    runner.main()
    batch = next(out.iterdir())
    completion = json.loads((batch/'completion.json').read_text())
    assert completion == dict(complete=True, expected=2, completed=2, training=False)
    assert (batch/'prepared/设备27/hourly.csv').exists()
    plan = json.loads((batch/'plan.json').read_text())
    assert [job['length'] for job in plan['jobs']] == [6, 12]
    assert all(job['model'] == 'Informer' for job in plan['jobs'])
    assert not list(batch.rglob('*.pt'))
