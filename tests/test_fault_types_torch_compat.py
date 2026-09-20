"""Runtime compatibility checks: inference/serialization only, no training."""
import io
import pytest
import torch
from models.SAFE_FaultTypes import FaultTypeModel


@pytest.mark.parametrize('name', ['iTransformer', 'Informer', 'PatchTST'])
@pytest.mark.parametrize('length', [3, 24])
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
