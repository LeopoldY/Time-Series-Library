import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from data_provider.safe_cluster_data import ClusterDataset
from data_provider.paper2016_split import row_roles
from scripts.tools.summarize_cluster_cv import summarize


class ClusterCVTests(unittest.TestCase):
    def frame(self):
        frames = []
        for device in [27, 58]:
            dates = pd.date_range('2020', periods=1000, freq='h').to_numpy()
            dates[500:] += np.timedelta64(400, 'h')
            frames.append(pd.DataFrame({'device_id': device, 'date': dates,
                **{f'w_level{i}': np.full(1000, device) for i in range(1, 5)}}))
        return pd.concat(frames, ignore_index=True)

    def test_device_identity_gap_and_split_isolation(self):
        frame = self.frame()
        used = {}
        for split in ['train', 'val', 'test']:
            data = ClusterDataset(frame, 24, split, 4)
            used[split] = set()
            self.assertEqual(len(data.summary['devices']), 2)
            for i, target in enumerate(data.targets):
                history, marks, decoder, label, index, counts = data[i]
                device = data.device_ids[target]
                self.assertTrue((history.numpy() == device).all())
                self.assertTrue((counts.numpy() == device).all())
                self.assertEqual(data.dates[target]-data.dates[target-24], pd.Timedelta(hours=24))
                self.assertTrue((row_roles(1000, 4)[target % 1000 - 24:target % 1000 + 1] == split).all())
                used[split].update(range(target-24, target+1))
        self.assertFalse(used['train'] & used['test'])
        self.assertFalse(used['train'] & used['val'])
        self.assertFalse(used['val'] & used['test'])

    def test_single_class_device_is_kept_in_pool(self):
        frame = self.frame()
        frame.loc[frame.device_id == 27, [f'w_level{i}' for i in range(1, 5)]] = 0
        data = ClusterDataset(frame, 3, 'train', 0)
        self.assertGreater(data.summary['positive'], 0)
        self.assertGreater(data.summary['negative'], 0)
        self.assertEqual(len(data.summary['devices']), 2)

    def test_routing_ignores_validation_and_test_values(self):
        from run_safe_joint_routed_training import cluster_devices
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            rng = np.random.default_rng(42)
            for device, rate in enumerate([.03, .04, .3, .4, 1.5, 2.]):
                values = rng.poisson(rate, (300, 4))
                frame = pd.DataFrame(values, columns=[f'w_level{i}' for i in range(1, 5)])
                frame.insert(0, 'date', pd.date_range('2020', periods=300, freq='h'))
                path = root / f'{device}.csv'; frame.to_csv(path, index=False)
                paths[device] = path
            before = cluster_devices(paths, root / 'before', 0)
            for path in paths.values():
                frame = pd.read_csv(path)
                frame.loc[row_roles(len(frame), 0) != 'train', frame.columns[1:]] = 100000
                frame.to_csv(path, index=False)
            after = cluster_devices(paths, root / 'after', 0)
            self.assertEqual(before, after)
            self.assertEqual((root / 'before/device_features.csv').read_bytes(),
                             (root / 'after/device_features.csv').read_bytes())

    def test_summary_equal_fold_mean_and_missing_fold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            for fold in range(10):
                job = dict(id=f'f{fold}', model='Informer', seq_len=3, fold=fold,
                           audit={'test': {'windows': fold + 1}})
                jobs.append(job)
                folder = root / 'jobs' / job['id']; folder.mkdir(parents=True)
                summary = {'model': 'Informer', 'seq_len': 3, 'metrics': {'test_mse': float(fold),
                           'test_mae': float(fold), 'classification': {'f1': .5, 'roc_auc': None if fold == 0 else .5}}}
                path = folder / 'summary.json'; path.write_text(json.dumps(summary))
                if fold < 9:
                    (folder / 'complete.json').write_text(json.dumps({'run': str(folder.relative_to(root)),
                        'artifacts': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}}))
            (root / 'plan.json').write_text(json.dumps({'jobs': jobs}))
            self.assertFalse(summarize(root)['all_planned_jobs_complete'])
            self.assertTrue(pd.isna(pd.read_csv(root / 'cv_summary.csv').test_mse_mean.iloc[0]))
            folder = root / 'jobs/f9'; path = folder / 'summary.json'
            (folder / 'complete.json').write_text(json.dumps({'run': 'jobs/f9', 'artifacts': {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}}))
            self.assertTrue(summarize(root)['all_planned_jobs_complete'])
            row = pd.read_csv(root / 'cv_summary.csv').iloc[0]
            self.assertAlmostEqual(row.test_mse_mean, 4.5)
            self.assertAlmostEqual(row.test_mse_std, np.std(range(10), ddof=1))
            self.assertEqual(row.roc_auc_valid_folds, 9)
            self.assertTrue(pd.isna(row.roc_auc_mean))


if __name__ == '__main__': unittest.main()
