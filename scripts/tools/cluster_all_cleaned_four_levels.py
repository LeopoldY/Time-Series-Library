"""Reproduce four-level SAFE profiling and model routing for all cleaned devices."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
LEVELS = [f'w_level{i}' for i in range(1, 5)]
FEATURES = [f'Level{i}_{metric}' for i in range(1, 5) for metric in ['Sparsity', 'Mean_Intensity', 'CV']]
MODELS = {'Dense': 'Informer', 'Intermittent': 'iTransformer', 'Sparse': 'PatchTST'}


def run(source, output):
    files = sorted(source.glob('设备*_cleaned.csv'), key=lambda p: int(p.stem.split('_')[0][2:]))
    if len(files) < 3:
        raise ValueError('At least three device files required')
    records = []
    for path in files:
        data = pd.read_csv(path)
        assert list(data.columns) == ['date'] + LEVELS, path
        x = data[LEVELS].to_numpy(dtype=float)
        assert len(x) and np.isfinite(x).all() and (x >= 0).all() and (x == np.floor(x)).all(), path
        dates = pd.to_datetime(data.date)
        assert dates.is_unique and dates.is_monotonic_increasing, path
        assert dates.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(), path
        rec = dict(Device_Name=path.stem.split('_')[0], Device_ID=int(path.stem.split('_')[0][2:]),
                   File_Path=str(path.resolve()), Root_Path=str(path.parent.resolve()), Data_Path=path.name,
                   Rows=len(x), Start=data.date.iloc[0], End=data.date.iloc[-1],
                   All_Levels_Zero_Ratio=float((x == 0).all(axis=1).mean()),
                   Total_Events=int(x.sum()), Input_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        for i in range(4):
            mean = float(x[:, i].mean())
            rec.update({f'Level{i+1}_Sparsity': float((x[:, i] == 0).mean()),
                        f'Level{i+1}_Mean_Intensity': mean,
                        f'Level{i+1}_CV': float(x[:, i].std(ddof=0)/mean) if mean > 1e-6 else 0.0})
        rec['Sparsity'] = float(np.mean([rec[f'Level{i}_Sparsity'] for i in range(1, 5)]))
        records.append(rec)
    df = pd.DataFrame(records)
    assert df.Device_ID.is_unique
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    estimator = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = estimator.fit_predict(scaled)
    assert df.Cluster.nunique() == 3
    order = df.groupby('Cluster').Sparsity.mean().sort_values(kind='stable').index.tolist()
    labels = dict(zip(order, MODELS))
    df['Label'] = df.Cluster.map(labels)
    df['Model'] = df.Label.map(MODELS)
    assert df.Model.notna().all() and len(df) == len(files)
    assert np.array_equal(estimator.predict(scaled), df.Cluster.to_numpy())
    summary = []
    for cluster in order:
        subset = df[df.Cluster == cluster]
        summary.append(dict(Cluster=int(cluster), Label=labels[cluster], Model=MODELS[labels[cluster]],
                            Devices=len(subset), Mean_Sparsity=float(subset.Sparsity.mean()),
                            Mean_All_Levels_Zero_Ratio=float(subset.All_Levels_Zero_Ratio.mean()),
                            Device_IDs=','.join(map(str, subset.Device_ID.tolist()))))
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output/'device_model_assignments.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(output/'cluster_summary.csv', index=False, encoding='utf-8-sig')
    centers = pd.DataFrame(scaler.inverse_transform(estimator.cluster_centers_), columns=FEATURES)
    centers.insert(0, 'Cluster', range(3))
    centers['Label'] = centers.Cluster.map(labels)
    centers['Model'] = centers.Label.map(MODELS)
    centers.to_csv(output/'cluster_centers.csv', index=False, encoding='utf-8-sig')
    metadata = dict(input_directory=str(source.resolve()), devices=len(df), total_rows=int(df.Rows.sum()),
                    total_events=int(df.Total_Events.sum()), feature_columns=FEATURES,
                    n_clusters=3, random_state=42, n_init=10, input_order='numeric device ID ascending',
                    scope='all rows of all input devices; descriptive full-history routing',
                    routing='ascending cluster mean of four per-level zero ratios', mapping=MODELS,
                    cluster_labels=labels, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                    centers_scaled=estimator.cluster_centers_.tolist(), inertia=float(estimator.inertia_),
                    silhouette_score=float(silhouette_score(scaled, df.Cluster)),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__))
    (output/'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'device_model_mapping.json').write_text(json.dumps(dict(zip(df.Device_Name, df.Model)), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = ['# 清洗后全设备四级计数聚类与模型分配', '',
              f'输入：`{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`。共{len(df)}台设备、{df.Rows.sum():,}行小时数据、{df.Total_Events.sum():,}条告警。', '',
              '## 方法', '',
              '沿用既有四级12维口径：各级独立计算零值比例、均值、总体标准差/均值（均值≤1e-6时CV=0），StandardScaler后KMeans(n_clusters=3, random_state=42, n_init=10)。设备按数字编号升序输入，无设备过滤。', '',
              '按簇内四级零值比例平均值升序命名Dense、Intermittent、Sparse，对应Informer、iTransformer、PatchTST。类别表示本轮集合内相对位置；全级同时零值比例仅作描述。', '',
              '## 分配结果', '', '| 类别 | 模型 | 设备数 | 平均零值比例 | 设备编号 |', '| --- | --- | ---: | ---: | --- |']
    for row in summary:
        report.append(f"| {row['Label']} | {row['Model']} | {row['Devices']} | {row['Mean_Sparsity']:.6f} | {row['Device_IDs']} |")
    report += ['', '## 验证与解释', '',
               '- 验证输入四级计数非负、有限且为整数，时间唯一且连续；全部设备恰好分配一个模型，分配结果与最近聚类中心预测一致。',
               '- 本次使用每台设备首末告警之间的完整小时序列，包括补零小时；未再次删除长零区间。与旧80设备或四设备聚类的数据范围不同，不能直接将分配变化归因于清洗。',
               '- 全量时间段聚类用于设备画像。正式预测评估应仅在训练时间段提取特征、拟合标准化器及聚类，避免未来信息影响模型选择。',
               '- 这是既定聚类路由规则，不是模型性能择优结果；本次未训练预测模型。', '',
               '## 文件与复现', '',
               '- device_model_assignments.csv：逐设备12维特征、类别、模型、输入路径及SHA256。',
               '- cluster_summary.csv、cluster_centers.csv：簇汇总与原始特征空间中心。',
               '- device_model_mapping.json：设备到模型的映射。',
               '- run_metadata.json：参数、标准化统计、中心、软件版本与聚类质量指标。',
               '- 复现：安装numpy、pandas、scikit-learn后，在项目目录运行 `python3 scripts/tools/cluster_all_cleaned_four_levels.py`。', '']
    (output/'README.md').write_text('\n'.join(report), encoding='utf-8')
    saved = pd.read_csv(output/'device_model_assignments.csv')
    assert saved.Device_ID.tolist() == df.Device_ID.tolist() and saved.Model.tolist() == df.Model.tolist()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Validated {len(saved)} devices; silhouette={metadata["silhouette_score"]:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'dataset/fault_raw/processed_all_cleaned/four_levels')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/safe_all_cleaned_four_levels_clustering')
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
