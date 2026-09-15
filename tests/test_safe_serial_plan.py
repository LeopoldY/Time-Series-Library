"""Serial plan routing, single-class handling and source integrity."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.tools.run_routed_two_stage_serial import build_plan
from run_safe_two_stage import parser
from run_safe_binary import configure


class SerialPlanTests(unittest.TestCase):
    def make_data(self, root):
        rows=[]
        for device,model in [(0,'Informer'),(1,'PatchTST')]:
            x=np.zeros((100,4),dtype=int)
            if device==0: x[::5,3]=1
            frame=pd.DataFrame(x,columns=[f'w_level{i}' for i in range(1,5)])
            frame.insert(0,'date',pd.date_range('2020-01-01',periods=100,freq='h'))
            p=root/f'设备{device}_cleaned.csv';frame.to_csv(p,index=False)
            rows.append(dict(Device_Name=f'设备{device}',Device_ID=device,Model=model,Input_SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
        route=root/'routes.csv'
        with route.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return route

    def test_plan_order_routes_options_and_single_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);route=self.make_data(root)
            p=build_plan(root,route,[3,24],[2021,2022],'cpu',root/'out')
            self.assertEqual((p['regression_jobs'],p['classification_jobs']),(8,4))
            self.assertEqual([j['device'] for j in p['jobs']],['设备0']*4+['设备1']*4)
            for job in p['jobs']:
                args=configure(parser().parse_args(job['options']))
                self.assertEqual(args.model,job['model'])
                self.assertEqual(args.stage,'both' if job['device']=='设备0' else 'regression')
                self.assertEqual((args.head_type,args.head_loss,args.head_selection),('history','asymmetric','ap'))
                self.assertFalse(args.train)

    def test_changed_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);route=self.make_data(root)
            with (root/'设备0_cleaned.csv').open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'differs'):
                build_plan(root,route,[24],[2021],'cpu',root/'out')

    def test_missing_device_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);route=self.make_data(root);(root/'设备1_cleaned.csv').unlink()
            with self.assertRaisesRegex(ValueError,'coverage'):
                build_plan(root,route,[24],[2021],'cpu',root/'out')


if __name__=='__main__':
    unittest.main()
