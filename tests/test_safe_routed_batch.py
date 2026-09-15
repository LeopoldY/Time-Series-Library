import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from run_safe_routed_batch import build_plan, command, digest, summarize


class RoutedBatchTests(unittest.TestCase):
    def test_plan_selects_validation_and_fails_on_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir()
            (data/'设备27_cleaned.csv').write_text('date,w_level1,w_level2,w_level3,w_level4\n')
            routes=root/'routes.csv';routes.write_text('Device_Name,Model,Label\n设备27,iTransformer,Intermittent\n')
            checkpoints={}
            for stamp,score in [('old',.2),('new',.7)]:
                run=root/'checkpoints'/'device27_iTransformer_L3'/stamp
                (run/'stage1_regression').mkdir(parents=True)
                cfg={'device_id':'27','model':'iTransformer','seq_len':3,'regression_loss':'mse',
                     'input_sha256':digest(data/'设备27_cleaned.csv'),
                     'data_protocol':'raw_counts_gap_safe_70_10_20_v1',
                     'feature_order':['w_level1','w_level2','w_level3','w_level4']}
                (run/'config.json').write_text(json.dumps(cfg))
                path=run/'stage1_regression/best_backbone.pt';path.write_text(stamp)
                checkpoints[str(path)]={'stage':'regression','config':cfg,'backbone':{},'validation':{'mse':score}}
            with patch('run_safe_routed_batch.read_checkpoint',side_effect=lambda p:checkpoints[str(p)]):
                plan=build_plan(routes,root/'checkpoints',data,[3],'mse')
                self.assertIn('/old/',plan['jobs'][0]['checkpoint'])
                self.assertEqual(plan['jobs'][0]['candidate_count'],2)
                with self.assertRaisesRegex(ValueError,'L6'):
                    build_plan(routes,root/'checkpoints',data,[3,6],'mse')

    def test_command_preserves_architecture_and_head_only(self):
        job={'device_id':'27','model':'Informer','seq_len':12,'checkpoint':'/tmp/with space/best.pt',
             'data_path':'/tmp/data/设备27_cleaned.csv','config':{'d_model':64,'n_heads':4,'d_ff':128,
             'seed':42,'regression_loss':'mae','batch_size':8}}
        args=SimpleNamespace(batch_size=0,epochs=3,patience=2,head_learning_rate=.001,threshold=.6,device='cpu')
        cmd=command(job,Path('/tmp/out'),args)
        self.assertEqual(cmd[cmd.index('--model')+1],'Informer')
        self.assertEqual(cmd[cmd.index('--stage')+1],'head')
        self.assertEqual(cmd[cmd.index('--d-model')+1],'64')
        self.assertNotIn('--train',cmd)

    def test_summary_recomputes_predictions_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';(run/'stage2_head').mkdir(parents=True)
            job={'id':'j1','device_id':'27','model':'Informer','seq_len':3,'checkpoint':'source.pt','validation_score':.2}
            plan={'jobs':[job],'devices':['27'],'lengths':[3],'head_options':{'threshold':.5}}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'state.json').write_text(json.dumps({'j1':{'status':'complete','run_dir':str(run)}}))
            p=run/'stage2_head/predictions.csv'
            p.write_text('device_id,model,seq_len,label_rule,target_time,y_true,y_pred,probability,threshold\n'
                         '27,Informer,3,any_level_next_hour,2020-01-01 00:00:00,1,1,0.8,0.5\n'
                         '27,Informer,3,any_level_next_hour,2020-01-01 01:00:00,0,1,0.6,0.5\n')
            self.assertTrue(summarize(root))
            with (root/'all_metrics.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            self.assertEqual(float(row['accuracy']),.5)
            self.assertAlmostEqual(float(row['f1']),2/3)
            plan['jobs'].append({**job,'id':'j2'})
            (root/'plan.json').write_text(json.dumps(plan))
            self.assertFalse(summarize(root))
            self.assertEqual(json.loads((root/'completion.json').read_text())['missing'],['j2'])


if __name__=='__main__':unittest.main()
