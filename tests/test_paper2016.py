import unittest
import numpy as np
import pandas as pd
from scripts.tools.prepare_paper2016 import clean_events
from data_provider.paper2016_split import split_targets, row_roles

class PaperTests(unittest.TestCase):
    def test_thresholds_and_anchor(self):
        for level, threshold in zip(['提示','次要','重要','紧急'],[900,240,180,120]):
            frame=pd.DataFrame([{'告警源':'设备1','级别':level,'名称':'a','告警类型':kind,'empty':' ',
                'Repair_Interval':duration,'发生时间':(pd.Timestamp('2020-01-01')+pd.Timedelta(seconds=s)).strftime('%m/%d/%Y %H:%M:%S')}
                for s,duration,kind in [(0,22,'普通告警'),(threshold-1,22,'普通告警'),(threshold,22,'普通告警'),(threshold*2,21,'普通告警'),(threshold*3,22,'衍生告警')]])
            result,audit=clean_events(frame)
            self.assertEqual(len(result),2)
            self.assertEqual([audit[k] for k in ['derived_removed','flash_removed','duplicate_removed']], [1,1,1])
            self.assertNotIn('empty',result)
            self.assertEqual(len(clean_events(result.drop(columns='_time'))[0]),2)

    def test_disjoint_windows_and_gaps(self):
        dates=pd.date_range('2020',periods=1000,freq='h').to_numpy()
        dates[500:]+=np.timedelta64(400,'h')
        test_sets=[]
        for fold in range(10):
            sets=[]
            for role in ['train','val','test']:
                targets=split_targets(dates,24,role,fold); used=set()
                for t in targets:
                    self.assertTrue((row_roles(len(dates),fold)[t-24:t+1]==role).all())
                    self.assertEqual(dates[t]-dates[t-24],np.timedelta64(24,'h'))
                    used.update(range(t-24,t+1))
                sets.append(used)
                if role=='test': test_sets.append(set(targets))
            for i in range(3):
                for j in range(i): self.assertFalse(sets[i]&sets[j])
        for i in range(10):
            for j in range(i): self.assertFalse(test_sets[i]&test_sets[j])

if __name__=='__main__': unittest.main()
