"""No training: only synthetic batches and gradients, no optimizer steps."""
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from data_provider.safe_binary_data import SafeBinaryDataset, FEATURES
from models.SAFE_Binary import SafeBinaryModel
from run_safe_binary import parser, configure, metrics


class SafeBinaryTests(unittest.TestCase):
    def test_labels_splits_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.csv'
            dates = list(pd.date_range('2020-01-01', periods=100, freq='h'))
            dates[40:] = [t+pd.Timedelta(days=2) for t in dates[40:]]
            counts = np.zeros((100, 4), dtype=int)
            for t in range(100):
                if t % 5 != 4:
                    counts[t, t % 5] = 2
            df = pd.DataFrame(counts, columns=FEATURES); df.insert(0, 'date', dates)
            df.to_csv(path, index=False)
            parts = {s: SafeBinaryDataset(path, 3, s) for s in ['train', 'val', 'test']}
            self.assertEqual(parts['train'].summary['gap_rejected'], 3)
            self.assertEqual(parts['val'].targets[0], 70)
            self.assertEqual(parts['test'].targets[0], 80)
            for d in parts.values():
                for j in range(len(d)):
                    x, _, dm, y, t = d[j]
                    np.testing.assert_array_equal(x.numpy(), counts[t-3:t])
                    self.assertEqual(y.item(), float(counts[t].sum()>0))
                    self.assertEqual(dm.shape, (2, 4))
                    self.assertTrue(all(dates[k]-dates[k-1] == pd.Timedelta(hours=1)
                                        for k in range(t-2, t+1)))
            self.assertTrue(set(parts['train'].targets).isdisjoint(parts['val'].targets))

    def test_backbones_gradients_and_checkpoint(self):
        torch.set_num_threads(1)
        for device in ['27', '58', '83']:
            for length in [3, 6, 12, 24]:
                with self.subTest(device=device, length=length):
                    args = configure(parser().parse_args(['--device-id', device, '--seq-len', str(length),
                                                         '--d-model', '16', '--n-heads', '2', '--d-ff', '32']))
                    torch.manual_seed(7)
                    model = SafeBinaryModel(args)
                    self.assertEqual(sum(p.numel() for p in model.head.parameters()), 97)
                    x = torch.rand(2, length, 4)
                    xm, dm = torch.rand(2, length, 4), torch.rand(2, 2, 4)
                    seen = []
                    hook = model.backbone.register_forward_pre_hook(lambda m, a: seen.append(a[2].detach().clone()))
                    logits = model(x, xm, dm); hook.remove()
                    self.assertEqual(tuple(logits.shape), (2, 1))
                    torch.testing.assert_close(seen[0][:, 0], x[:, -1])
                    self.assertEqual(torch.count_nonzero(seen[0][:, 1]).item(), 0)
                    nn.BCEWithLogitsLoss()(logits, torch.tensor([[0.], [1.]])).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.head.parameters()))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters()))
                    buffer = io.BytesIO(); torch.save({'model': model.state_dict()}, buffer); buffer.seek(0)
                    clone = SafeBinaryModel(args)
                    clone.load_state_dict(torch.load(buffer, weights_only=True)['model'])
                    model.eval(); clone.eval()
                    # ProbAttention samples in eval too: align RNG for roundtrip comparison.
                    torch.manual_seed(13); expected = model(x, xm, dm)
                    torch.manual_seed(13); actual = clone(x, xm, dm)
                    torch.testing.assert_close(actual, expected)

    def test_metrics_single_class(self):
        result = metrics(np.zeros(4, dtype=int), np.array([0, 1, 0, 0]), np.array([.1, .8, .2, .3]))
        self.assertEqual(result['fp'], 1)
        for key in ['recall', 'f1', 'average_precision', 'roc_auc']:
            self.assertIsNone(result[key])
        self.assertEqual(metrics(np.array([1, 0]), np.array([0, 0]))['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
