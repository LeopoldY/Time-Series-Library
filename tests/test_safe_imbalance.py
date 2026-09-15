"""Imbalance head: loss math, threshold selection, and protocol preservation."""
import tempfile
import io
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from models.SAFE_Imbalance import HistoryAwareHead, AsymmetricBinaryLoss, select_threshold
from models.SAFE_Binary import SafeBinaryModel
from run_safe_binary import configure
from run_safe_two_stage import parser, state_hash, TwoStageDataset
from data_provider.safe_binary_data import FEATURES


class ImbalanceTests(unittest.TestCase):
    def test_asl_reduces_to_bce_and_extreme_gradients(self):
        x = torch.tensor([[-1000.], [1000.], [-2.], [2.]], requires_grad=True)
        y = torch.tensor([[1.], [0.], [0.], [1.]])
        torch.testing.assert_close(AsymmetricBinaryLoss(0, 0, 0)(x, y), nn.BCEWithLogitsLoss()(x, y))
        loss = AsymmetricBinaryLoss()(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(x.grad[0].item(), 0)  # missed positive remains learnable

    def test_easy_negatives_suppressed_positive_unchanged(self):
        x = torch.tensor([[-4.]], requires_grad=True)
        zero, one = torch.zeros_like(x), torch.ones_like(x)
        loss = AsymmetricBinaryLoss()(x, zero)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(x.grad.item(), 0.)
        torch.testing.assert_close(AsymmetricBinaryLoss()(x, one), nn.BCEWithLogitsLoss()(x, one))
        x = torch.tensor([[0.]], requires_grad=True)
        AsymmetricBinaryLoss()(x, torch.zeros_like(x)).backward()
        self.assertGreater(x.grad.item(), 0.)

    def test_threshold_matches_brute_force_and_handles_ties(self):
        y = np.array([1,0,1,0,0,1]); p = np.array([.8,.8,.4,.2,.1,.1])
        for beta in [1., 2.]:
            result = select_threshold(y,p,'val_fbeta',beta=beta)
            def score(t):
                pred = p >= t; tp = sum(pred & (y==1)); fp = sum(pred & (y==0)); fn = sum(~pred & (y==1))
                return (1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)
            best = max(np.unique(p), key=lambda t:(score(t),t))
            self.assertEqual(result['threshold'],best)
        self.assertEqual(select_threshold([0,0],[.1,.9],'val_fbeta',.6)['threshold'],.6)
        self.assertEqual(select_threshold([1,1],[.1,.9],'val_fbeta')['source'],'fixed')
        with self.assertRaises(ValueError):
            select_threshold([1],[float('nan')])

    def test_threshold_csv_float32_tie(self):
        score = np.array([.1450362503528595, .1], dtype=np.float32).astype(np.float64)
        decision = select_threshold([1,0], score, 'val_fbeta')
        frame = pd.DataFrame(dict(probability=score, threshold=decision['threshold'],
                                  y_pred=(score>=decision['threshold']).astype(int)))
        saved = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        self.assertTrue(saved.y_pred.eq((saved.probability>=saved.threshold).astype(int)).all())

    def test_history_features_and_future_label_protocol(self):
        x = torch.zeros(1,3,4); x[0,0,0] = 2; x[0,-1,3] = 1
        features = HistoryAwareHead.history_features(x)
        torch.testing.assert_close(features[0,16:],torch.tensor([2/3,1.,1.,0.]))
        self.assertEqual(features.shape,(1,20))
        with tempfile.TemporaryDirectory() as tmp:
            counts = np.zeros((100,4)); counts[10,3] = 1; counts[11,0] = 1
            frame = pd.DataFrame(counts,columns=FEATURES)
            frame.insert(0,'date',pd.date_range('2020-01-01',periods=100,freq='h'))
            path = Path(tmp)/'data.csv';frame.to_csv(path,index=False)
            data = TwoStageDataset(path,3,'train')
            sample = data[np.flatnonzero(data.targets==10)[0]]
            self.assertEqual(sample[3].item(),1.)  # level4 alone is positive
            self.assertEqual(sample[0].sum().item(),0.)  # target not in history
            self.assertEqual(data.labels[11],1.)
            self.assertEqual(data.labels[12],0.)

    def test_all_backbones_freeze_optimizer_and_roundtrip(self):
        torch.set_num_threads(1)
        for name in ['Informer','iTransformer','PatchTST']:
            with self.subTest(model=name):
                args = configure(parser().parse_args(['--device-id','27','--model',name,'--seq-len','6',
                    '--d-model','16','--d-ff','32','--n-heads','2','--head-type','history']))
                model = SafeBinaryModel(args); model.freeze_backbone(); model.head.initialize_prior(5,95)
                before = state_hash(model.backbone)
                optimizer = torch.optim.Adam(model.head.parameters(),lr=.01)
                x, xm, dm = torch.rand(4,6,4),torch.rand(4,6,4),torch.rand(4,2,4)
                model.train(); loss = AsymmetricBinaryLoss()(model(x,xm,dm),torch.tensor([[0.],[0.],[0.],[1.]]))
                loss.backward(); optimizer.step()
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))
                self.assertEqual(state_hash(model.backbone),before)
                clone = SafeBinaryModel(args);clone.load_state_dict(model.state_dict());clone.freeze_backbone()
                model.eval();clone.eval();torch.manual_seed(5);expected=model(x,xm,dm)
                torch.manual_seed(5);torch.testing.assert_close(clone(x,xm,dm),expected)
                self.assertEqual(expected.shape,(4,1))


if __name__ == '__main__':
    unittest.main()
