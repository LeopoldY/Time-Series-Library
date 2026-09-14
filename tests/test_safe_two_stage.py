"""Two-stage contract checks: synthetic data, no training loops or optimizer steps."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_provider.safe_binary_data import FEATURES
from models.SAFE_Binary import SafeBinaryModel
from run_safe_binary import configure
from run_safe_two_stage import (TwoStageDataset, parser, state_hash, load_backbone,
                                run_epoch, COMPATIBILITY_KEYS)


class TwoStageTests(unittest.TestCase):
    def test_frozen_parameters_buffers_and_head_gradients(self):
        torch.set_num_threads(1)
        for name in ['Informer', 'iTransformer', 'PatchTST']:
            with self.subTest(model=name):
                args = configure(parser().parse_args(['--device-id', '27', '--model', name,
                        '--seq-len', '6', '--d-model', '16', '--d-ff', '32', '--n-heads', '2']))
                self.assertEqual(args.model, name)
                model = SafeBinaryModel(args)
                # Stage1 gradients reach backbone, not the unused head.
                x, xm, dm = torch.rand(2, 6, 4), torch.rand(2, 6, 4), torch.rand(2, 2, 4)
                nn.MSELoss()(model.forecast(x, xm, dm), torch.rand(2, 4)).backward()
                self.assertTrue(any(p.grad is not None for p in model.backbone.parameters()))
                self.assertTrue(all(p.grad is None for p in model.head.parameters()))
                model.freeze_backbone()
                before = state_hash(model.backbone)
                for _ in range(2):
                    model.train()
                    self.assertFalse(model.backbone.training)
                    self.assertTrue(model.head.training)
                    self.assertTrue(all(not m.training for m in model.backbone.modules()))
                    nn.BCEWithLogitsLoss()(model(x, xm, dm), torch.tensor([[0.], [1.]])).backward()
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.backbone.parameters()))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))
                self.assertEqual(before, state_hash(model.backbone))

    def test_regression_targets_and_mse_mae(self):
        class FixedForecast(nn.Module):
            def forecast(self, x, xm, dm):
                # Next count is last+1; errors are [0,1,2,3].
                return x[:, -1] + torch.tensor([1., 2., 3., 4.])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'data.csv'
            df = pd.DataFrame(np.repeat(np.arange(100)[:, None], 4, axis=1), columns=FEATURES)
            df.insert(0, 'date', pd.date_range('2020-01-01', periods=100, freq='h'))
            df.to_csv(path, index=False)
            data = TwoStageDataset(path, 3, 'test')
            x, xm, dm, binary, t, target = data[0]
            np.testing.assert_array_equal(target.numpy(), [t]*4)
            self.assertEqual(binary.item(), 1.)
            # Batch size 7 checks correct weighting of the final short batch.
            stats, true, pred, indices = run_epoch(FixedForecast(), DataLoader(data, batch_size=7),
                                                   'regression', nn.MSELoss(), torch.device('cpu'), collect=True)
            self.assertAlmostEqual(stats['mse'], 3.5)
            self.assertAlmostEqual(stats['mae'], 1.5)
            self.assertAlmostEqual(stats['loss'], 3.5)
            self.assertEqual(stats['per_level']['w_level4']['mae'], 3.)
            np.testing.assert_array_equal(indices, data.targets)

    def test_checkpoint_compatibility_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            args = configure(parser().parse_args(['--device-id', '83', '--seq-len', '3',
                                                  '--d-model', '16', '--d-ff', '32', '--n-heads', '2']))
            model = SafeBinaryModel(args)
            config = {key: vars(args).get(key) for key in COMPATIBILITY_KEYS}
            config.update(feature_order=FEATURES, input_sha256='synthetic', data_protocol='test')
            path = Path(temp)/'best_backbone.pt'
            checkpoint = {'stage': 'regression', 'config': config, 'backbone': model.backbone.state_dict(),
                          'backbone_sha256': state_hash(model.backbone)}
            torch.save(checkpoint, path)
            clone = SafeBinaryModel(args)
            load_backbone(clone, path, config, 'cpu')
            self.assertEqual(state_hash(clone.backbone), state_hash(model.backbone))
            with self.assertRaisesRegex(ValueError, 'input_sha256'):
                load_backbone(clone, path, {**config, 'input_sha256': 'wrong-data'}, 'cpu')
            with self.assertRaisesRegex(ValueError, 'seq_len'):
                load_backbone(clone, path, {**config, 'seq_len': 24}, 'cpu')
            checkpoint['backbone_sha256'] = 'wrong'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_backbone(clone, path, config, 'cpu')


if __name__ == '__main__':
    unittest.main()
