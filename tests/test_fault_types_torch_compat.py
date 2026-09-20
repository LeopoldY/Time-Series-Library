"""Runtime compatibility checks: inference/serialization only, no training."""
import io
import pytest
import torch
from models.SAFE_FaultTypes import FaultTypeModel


@pytest.mark.parametrize('name', ['iTransformer', 'Informer', 'PatchTST'])
@pytest.mark.parametrize('length', [6, 12])
def test_inference_and_safe_checkpoint(name, length):
    torch.set_num_threads(1)
    model = FaultTypeModel(name, length, 6, 2, 16).eval()
    checkpoint = dict(model=model.state_dict(), config=dict(model=name, seq_len=length,
                      learning_rate=.001, torch_version=str(torch.__version__)),
                      audit=dict(names=['type_A', 'type_B'], mean=[0., 1.], scale=[1., 2.]),
                      pos_weight=[1., 2., 3.], epoch=1, val_loss=.75,
                      thresholds=[.5, .25, .75], supported_types=[True, False])
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu', weights_only=True)
    assert restored['config'] == checkpoint['config']
    assert restored['audit'] == checkpoint['audit']
    assert restored['thresholds'] == checkpoint['thresholds']
    assert restored['val_loss'] == .75
    clone = FaultTypeModel(name, length, 6, 2, 16).eval()
    clone.load_state_dict(restored['model'], strict=True)
    with torch.inference_mode():
        logits = clone(torch.zeros(2, length, 6), torch.zeros(2, length, 4),
                       torch.zeros(2, 2, 4))
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert all(p.grad is None for p in clone.parameters())


@pytest.mark.parametrize('name', ['iTransformer', 'Informer', 'PatchTST'])
def test_frozen_backbone_mode_and_inference(name):
    from run_safe_fault_types import state_hash
    model = FaultTypeModel(name, 6, 6, 2, 16)
    model.freeze_backbone()
    model.train()
    assert model.head.training
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    before = state_hash(model.backbone)
    with torch.inference_mode():
        x, xm, dm = torch.zeros(2,6,6), torch.zeros(2,6,4), torch.zeros(2,2,4)
        assert model.forecast(x,xm,dm).shape == (2,6)
        assert model(x,xm,dm).shape == (2,3)
    assert state_hash(model.backbone) == before
    assert all(p.grad is None for p in model.parameters())


def test_default_ten_jobs_and_routes():
    from run_safe_fault_types import build_plan, DEFAULT_ROUTES
    plan = build_plan([27,28,39,46,58], [6,12], 2021, DEFAULT_ROUTES)
    assert len(plan) == 10
    expected = {27:'Informer', 28:'iTransformer', 39:'Informer', 46:'PatchTST', 58:'Informer'}
    assert {(j['device'], j['length']) for j in plan} == {(d,l) for d in expected for l in (6,12)}
    assert all(j['model'] == expected[j['device']] and j['seed'] == 2021 for j in plan)


def test_stage_parameter_scope_without_training(tmp_path, monkeypatch):
    import run_safe_fault_types as runner
    from torch import nn
    model = FaultTypeModel('iTransformer', 6, 6, 2, 16)
    calls = []
    def fake_epoch(model, loader, criterion, device, optimizer=None, stage='head'):
        # No forward/backward, optimizer step, or training data is used in this test.
        if optimizer is not None:
            actual = {id(p) for g in optimizer.param_groups for p in g['params']}
            module = model.backbone if stage == 'regression' else model.head
            assert actual == {id(p) for p in module.parameters()}
        calls.append((stage, optimizer is not None))
        return .5, None, None, None
    monkeypatch.setattr(runner, 'epoch', fake_epoch)
    before = runner.state_hash(model)
    loaders = {'train':None, 'val':None}
    runner.fit_stage(model, loaders, nn.MSELoss(), 'cpu', tmp_path/'regression',
                     {}, {}, 'regression', 1, 1, .0001)
    model.freeze_backbone()
    runner.fit_stage(model, loaders, nn.MSELoss(), 'cpu', tmp_path/'head',
                     {}, {}, 'head', 1, 1, .001)
    assert calls == [('regression', True), ('regression', False), ('head', True), ('head', False)]
    assert runner.state_hash(model) == before
    assert (tmp_path/'regression/best_backbone.pt').exists()
    assert (tmp_path/'head/best_head.pt').exists()
    assert (tmp_path/'head/freeze_audit.json').exists()
