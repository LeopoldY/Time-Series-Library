"""Rebuild raw hourly counts and apply the verified selected-device zero-run rule."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sklearn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'dataset/fault_raw/raw_data'
OUT = ROOT / 'dataset/fault_all_selected_cleaned'
REPORT = ROOT / 'docs/safe_all_selected_cleaned_clustering'
LEVELS = [f'w_level{i}' for i in range(1, 5)]
FEATURES = [f'Level{i}_{m}' for i in range(1, 5) for m in ['Sparsity', 'Mean_Intensity', 'CV']]
MAPPING = {'Dense': 'Informer', 'Intermittent': 'iTransformer', 'Sparse': 'PatchTST'}


def clean(frame):
    zero = frame[LEVELS].eq(0).all(axis=1)
    group = zero.ne(zero.shift()).cumsum()
    remove = zero & zero.groupby(group).transform('size').ge(336)
    return frame.loc[~remove].reset_index(drop=True), remove, group


def main():
    devices = []
    for path in sorted(SOURCE.glob('*.csv'), key=lambda p: int(p.name.split('_')[0][2:])):
        raw = pd.read_csv(path)
        times = pd.to_datetime(raw['发生时间'], format='%m/%d/%Y %H:%M:%S', errors='raise')
        levels = raw['级别'].map({'提示': 0, '次要': 1, '重要': 2, '紧急': 3})
        assert len(raw) and times.notna().all() and levels.notna().all(), path
        assert raw['告警源'].eq(path.name.split('_')[0]).all(), path
        devices.append((path, times, levels.astype(int)))
    start = min(t.min() for _, t, _ in devices)
    end = max(t.max() for _, t, _ in devices)
    dates = pd.date_range(start + pd.Timedelta(minutes=30), periods=int((end-start)//pd.Timedelta(hours=1))+1, freq='h')
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    audits, ranges, comparisons, differences, records = [], [], [], [], []
    for path, times, levels in devices:
        name = path.name.split('_')[0]
        x = np.zeros((len(dates), 4), dtype=np.int64)
        bins = ((times-start)//pd.Timedelta(hours=1)).to_numpy()
        np.add.at(x, (bins, levels.to_numpy()), 1)
        frame = pd.DataFrame(x, columns=LEVELS)
        frame.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
        cleaned, remove, group = clean(frame)
        assert len(cleaned) and int(cleaned[LEVELS].sum().sum()) == len(times)
        assert frame.loc[remove, LEVELS].eq(0).all().all()
        for _, block in frame.loc[remove].groupby(group[remove]):
            ranges.append(dict(Device_Name=name, Start=block.date.iloc[0], End=block.date.iloc[-1], Removed_Hours=len(block)))
        selected = ROOT / f'dataset/fault_selected/{name}.csv'
        reference = ROOT / f'dataset/fault_selected_cleaned/{name}_cleaned.csv'
        if reference.exists():
            old = pd.read_csv(selected)
            ref = pd.read_csv(reference)
            reproduced, _, _ = clean(old)
            assert reproduced.equals(ref), name
            assert frame.date.equals(old.date)
            delta = frame[LEVELS].to_numpy() - old[LEVELS].to_numpy()
            for i, j in np.argwhere(delta != 0):
                differences.append(dict(Device_Name=name, Date=frame.date.iloc[i], Level=LEVELS[j], Current_Raw_Count=int(x[i,j]), Legacy_Count=int(old.iloc[i,j+1]), Difference=int(delta[i,j])))
            comparisons.append(dict(Device_Name=name, Legacy_Rows=len(old), Reference_Rows=len(ref), Rule_Reproduces_Reference=True, Raw_Count_Different_Cells=int((delta!=0).sum()), Raw_Minus_Legacy_Events=int(delta.sum()), Current_Cleaned_Rows=len(cleaned), Current_Equals_Reference=cleaned.equals(ref)))
        target = OUT / f'{name}_cleaned.csv'
        cleaned.to_csv(target, index=False)
        pd.testing.assert_frame_equal(pd.read_csv(target), cleaned)
        assert clean(cleaned)[0].equals(cleaned)
        gaps = int(pd.to_datetime(cleaned.date).diff().iloc[1:].ne(pd.Timedelta(hours=1)).sum())
        audits.append(dict(Device_Name=name, Raw_Events=len(times), Original_Hours=len(frame), Removed_Hours=int(remove.sum()), Cleaned_Hours=len(cleaned), Gap_Count=gaps, Source_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), Output_SHA256=hashlib.sha256(target.read_bytes()).hexdigest()))
        rec = dict(Device_Name=name, Device_ID=int(name[2:]), File_Path=str(target), Root_Path=str(OUT), Data_Path=target.name, Rows=len(cleaned), Total_Events=len(times), Gap_Count=gaps, Input_SHA256=hashlib.sha256(target.read_bytes()).hexdigest())
        arr = cleaned[LEVELS].to_numpy()
        for i in range(4):
            mean = float(arr[:,i].mean())
            rec.update({f'Level{i+1}_Sparsity': float((arr[:,i]==0).mean()), f'Level{i+1}_Mean_Intensity': mean, f'Level{i+1}_CV': float(arr[:,i].std()/mean) if mean>1e-6 else 0.0})
        rec['Sparsity'] = np.mean([rec[f'Level{i}_Sparsity'] for i in range(1,5)])
        records.append(rec)
    df = pd.DataFrame(records)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(scaled)
    order = df.groupby('Cluster').Sparsity.mean().sort_values(kind='stable').index
    labels = dict(zip(order, MAPPING))
    df['Label'] = df.Cluster.map(labels)
    df['Model'] = df.Label.map(MAPPING)
    assert df.Device_ID.is_unique and df.Model.notna().all() and len(df)==len(devices)
    assert np.array_equal(km.predict(scaled), df.Cluster)
    summaries = [dict(Cluster=int(c), Label=labels[c], Model=MAPPING[labels[c]], Devices=int((df.Cluster==c).sum()), Mean_Sparsity=float(df.loc[df.Cluster==c,'Sparsity'].mean()), Device_IDs=','.join(df.loc[df.Cluster==c,'Device_ID'].astype(str))) for c in order]
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
    centers.insert(0, 'Cluster', range(3))
    centers['Label'] = centers.Cluster.map(labels)
    for filename, table in [('device_model_assignments',df), ('cleaning_audit',pd.DataFrame(audits)), ('removed_intervals',pd.DataFrame(ranges)), ('reference_comparison',pd.DataFrame(comparisons)), ('legacy_count_differences',pd.DataFrame(differences)), ('cluster_summary',pd.DataFrame(summaries)), ('cluster_centers',centers)]:
        table.to_csv(REPORT/f'{filename}.csv',index=False,encoding='utf-8-sig')
    metadata = dict(source=str(SOURCE), output=str(OUT), devices=len(devices), raw_events=sum(a['Raw_Events'] for a in audits), original_hours=sum(a['Original_Hours'] for a in audits), removed_hours=sum(a['Removed_Hours'] for a in audits), cleaned_hours=sum(a['Cleaned_Hours'] for a in audits), start=str(start), last_event=str(end), first_label=str(dates[0]), last_label=str(dates[-1]), threshold_hours=336, feature_columns=FEATURES, n_clusters=3, random_state=42, n_init=10, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), centers_scaled=km.cluster_centers_.tolist(), silhouette_score=float(silhouette_score(scaled,df.Cluster)), versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (REPORT/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (REPORT/'device_model_mapping.json').write_text(json.dumps(dict(zip(df.Device_Name,df.Model)),ensure_ascii=False,indent=2)+'\n')
    lines = ['# 全部原始设备按四样本口径清洗、聚类与路由', '', '## 清洗口径与核验', '', '- 输入直接读取 fault_raw/raw_data 全部CSV，保留每条告警（不去重、不按名称或修复时长过滤）。空列不参与四级计数。', '- 复现 fault/get_data.py 的窗口定义：全设备最早发生时间为统一起点，左闭右开1小时窗口，date为窗口中点；全设备统一覆盖至最晚事件所在小时，缺失小时四级补零。', '- 复现 scripts/tools/clean_no_fault.py：四级同时为零且连续至少336小时，删除整个区间（含首尾），保留原始时间戳和所有非零计数。', '- 四台旧 fault_selected 数据应用此规则后，与 fault_selected_cleaned 逐行完全一致。设备27、69、83从当前原始事件重建后也完全一致；设备58当前原始事件比旧小时表多15条、分布在15个计数单元，这个差异发生于清洗前。没有证据说明这15条应被删除，本次完整保留。详细差异见 legacy_count_differences.csv。', '', '## 全量结果', '', f"共{metadata['devices']}台，{metadata['raw_events']:,}条告警；原始{metadata['original_hours']:,}小时行，删除{metadata['removed_hours']:,}个全零小时，保留{metadata['cleaned_hours']:,}行。全部文件回读验证，清洗前后告警总数守恒，清洗幂等，所有设备均完成路由。", '', '## 聚类与路由', '', '各级零值比例、均值、总体标准差/均值共12维；均值≤1e-6时CV=0。StandardScaler + KMeans(3, random_state=42, n_init=10)，按设备数字编号排序。按簇平均四级零值比例由低到高命名并映射模型。', '', '| 类别 | 模型 | 设备数 | 平均零值比例 | 设备编号 |', '| --- | --- | ---: | ---: | --- |']
    for s in summaries:
        lines.append(f"| {s['Label']} | {s['Model']} | {s['Devices']} | {s['Mean_Sparsity']:.6f} | {s['Device_IDs']} |")
    lines += ['', f"轮廓系数：{metadata['silhouette_score']:.6f}。", '', '## 使用与复现', '', '- 清洗数据：dataset/fault_all_selected_cleaned；逐设备结果：device_model_assignments.csv；模型映射：device_model_mapping.json。原始数据和四样本均未覆盖。', '- cleaning_audit.csv记录数量和输入输出SHA256；removed_intervals.csv记录每个删除区间；reference_comparison.csv记录四样本复现结果；run_metadata.json记录参数、版本与中心。', '- 此次为全历史设备画像和既定规则路由，没有训练模型或比较预测性能；正式评估应仅在训练段拟合聚类与标准化器。', '- 删除区间导致时间断点，训练滑窗必须按连续时间段划分，不能跨断点把相邻行视为相邻小时。', '- 复现：在项目根目录运行 python3 scripts/tools/clean_all_like_selected.py（需要numpy、pandas、scipy、scikit-learn）。', '']
    (REPORT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
