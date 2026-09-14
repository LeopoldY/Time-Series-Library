# SAFE 四级特征全设备聚类与模型分配

日期：2026-09-14。输入为 dataset/fault 全部80台设备的全量时间段。本轮在独立分析脚本中执行，未修改项目算法代码或原始数据，未启动模型训练。

## 特征与路由定义

每个级别分别计算零值比例 Sparsity、均值 Mean_Intensity、总体标准差与均值之比 CV，共12维；沿用原规则，均值不大于 1e-6 时 CV 取0。每维经 StandardScaler 标准化，KMeans(n_clusters=3, random_state=42, n_init=10)。不先把四级计数相加，以保留各级分布差异。

原路由使用四级告警的稀疏度为簇排序。本轮将排序指标明确扩展为每台设备四个级别零值比例的算术平均，再求簇均值；由低到高分配 Dense→Informer、Intermittent→iTransformer、Sparse→PatchTST。四级在特征构造和路由排序中均不设置额外严重性权重。

注意：四级零值比例的平均，不等于四级同时为0的窗口比例。后者另列 All_Levels_Zero_Ratio，仅用于描述，不参与本轮聚类或路由排序。类别名仅表示本轮集合内相对位置。

此次使用全量时间段做设备画像；后续正式预测评估应在训练段拟合特征及聚类，避免未来数据影响模型选择。

## 汇总

| 簇 | 类别 | 模型 | 设备数 | 平均路由稀疏度 | 平均无告警窗口比例 |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | Dense | Informer | 3 | 0.908394 | 0.691820 |
| 1 | Intermittent | iTransformer | 3 | 0.940146 | 0.796648 |
| 0 | Sparse | PatchTST | 74 | 0.989244 | 0.960647 |

与上一轮仅 w_level4 聚类相比，36 台设备模型分配改变。

## 全设备分配及比较

| 设备 | 类别 | 当前模型 | 上轮模型 | 是否改变 | 路由稀疏度 |
| --- | --- | --- | --- | --- | ---: |
| 设备0 | Sparse | PatchTST | iTransformer | 是 | 0.991719 |
| 设备1 | Sparse | PatchTST | PatchTST | 否 | 0.995467 |
| 设备2 | Sparse | PatchTST | iTransformer | 是 | 0.991057 |
| 设备3 | Sparse | PatchTST | iTransformer | 是 | 0.997207 |
| 设备4 | Sparse | PatchTST | PatchTST | 否 | 0.997893 |
| 设备5 | Sparse | PatchTST | PatchTST | 否 | 0.987946 |
| 设备6 | Sparse | PatchTST | PatchTST | 否 | 0.982213 |
| 设备7 | Sparse | PatchTST | PatchTST | 否 | 0.995443 |
| 设备8 | Sparse | PatchTST | PatchTST | 否 | 0.998015 |
| 设备9 | Sparse | PatchTST | PatchTST | 否 | 0.998358 |
| 设备10 | Sparse | PatchTST | PatchTST | 否 | 0.994414 |
| 设备11 | Sparse | PatchTST | PatchTST | 否 | 0.995296 |
| 设备12 | Sparse | PatchTST | iTransformer | 是 | 0.930958 |
| 设备13 | Sparse | PatchTST | iTransformer | 是 | 0.996790 |
| 设备14 | Sparse | PatchTST | iTransformer | 是 | 0.991449 |
| 设备15 | Sparse | PatchTST | PatchTST | 否 | 0.998750 |
| 设备16 | Sparse | PatchTST | PatchTST | 否 | 0.998579 |
| 设备17 | Sparse | PatchTST | PatchTST | 否 | 0.992919 |
| 设备18 | Sparse | PatchTST | PatchTST | 否 | 0.991792 |
| 设备19 | Sparse | PatchTST | iTransformer | 是 | 0.994267 |
| 设备20 | Sparse | PatchTST | PatchTST | 否 | 0.998187 |
| 设备21 | Sparse | PatchTST | PatchTST | 否 | 0.998726 |
| 设备23 | Sparse | PatchTST | iTransformer | 是 | 0.996178 |
| 设备24 | Sparse | PatchTST | iTransformer | 是 | 0.992258 |
| 设备25 | Sparse | PatchTST | iTransformer | 是 | 0.998407 |
| 设备26 | Sparse | PatchTST | PatchTST | 否 | 0.991376 |
| 设备27 | Dense | Informer | PatchTST | 是 | 0.919370 |
| 设备28 | Intermittent | iTransformer | PatchTST | 是 | 0.943895 |
| 设备29 | Sparse | PatchTST | PatchTST | 否 | 0.997354 |
| 设备30 | Sparse | PatchTST | PatchTST | 否 | 0.988362 |
| 设备31 | Sparse | PatchTST | PatchTST | 否 | 0.998652 |
| 设备32 | Sparse | PatchTST | PatchTST | 否 | 0.995761 |
| 设备33 | Sparse | PatchTST | PatchTST | 否 | 0.997893 |
| 设备34 | Sparse | PatchTST | PatchTST | 否 | 0.997403 |
| 设备35 | Sparse | PatchTST | PatchTST | 否 | 0.998309 |
| 设备36 | Dense | Informer | PatchTST | 是 | 0.902318 |
| 设备37 | Sparse | PatchTST | PatchTST | 否 | 0.982164 |
| 设备38 | Sparse | PatchTST | PatchTST | 否 | 0.998726 |
| 设备39 | Sparse | PatchTST | iTransformer | 是 | 0.942180 |
| 设备40 | Sparse | PatchTST | PatchTST | 否 | 0.998383 |
| 设备42 | Intermittent | iTransformer | iTransformer | 否 | 0.946492 |
| 设备43 | Sparse | PatchTST | PatchTST | 否 | 0.993311 |
| 设备44 | Sparse | PatchTST | iTransformer | 是 | 0.991719 |
| 设备45 | Sparse | PatchTST | iTransformer | 是 | 0.993164 |
| 设备46 | Sparse | PatchTST | PatchTST | 否 | 0.951220 |
| 设备47 | Sparse | PatchTST | PatchTST | 否 | 0.973932 |
| 设备48 | Sparse | PatchTST | PatchTST | 否 | 0.998946 |
| 设备49 | Sparse | PatchTST | iTransformer | 是 | 0.994732 |
| 设备50 | Sparse | PatchTST | iTransformer | 是 | 0.983854 |
| 设备51 | Sparse | PatchTST | iTransformer | 是 | 0.985177 |
| 设备52 | Sparse | PatchTST | iTransformer | 是 | 0.993458 |
| 设备53 | Sparse | PatchTST | iTransformer | 是 | 0.995051 |
| 设备54 | Sparse | PatchTST | iTransformer | 是 | 0.987015 |
| 设备55 | Sparse | PatchTST | iTransformer | 是 | 0.991106 |
| 设备56 | Sparse | PatchTST | PatchTST | 否 | 0.972535 |
| 设备57 | Sparse | PatchTST | iTransformer | 是 | 0.987995 |
| 设备58 | Dense | Informer | PatchTST | 是 | 0.903494 |
| 设备60 | Sparse | PatchTST | PatchTST | 否 | 0.971947 |
| 设备61 | Sparse | PatchTST | PatchTST | 否 | 0.994732 |
| 设备63 | Sparse | PatchTST | PatchTST | 否 | 0.987113 |
| 设备64 | Sparse | PatchTST | PatchTST | 否 | 0.988852 |
| 设备65 | Sparse | PatchTST | PatchTST | 否 | 0.991131 |
| 设备66 | Sparse | PatchTST | PatchTST | 否 | 0.994071 |
| 设备67 | Sparse | PatchTST | PatchTST | 否 | 0.987897 |
| 设备69 | Sparse | PatchTST | PatchTST | 否 | 0.982703 |
| 设备70 | Sparse | PatchTST | PatchTST | 否 | 0.987676 |
| 设备71 | Sparse | PatchTST | iTransformer | 是 | 0.981257 |
| 设备72 | Sparse | PatchTST | iTransformer | 是 | 0.983242 |
| 设备73 | Sparse | PatchTST | iTransformer | 是 | 0.981110 |
| 设备74 | Sparse | PatchTST | iTransformer | 是 | 0.989097 |
| 设备75 | Sparse | PatchTST | iTransformer | 是 | 0.994708 |
| 设备76 | Sparse | PatchTST | iTransformer | 是 | 0.995369 |
| 设备78 | Sparse | PatchTST | iTransformer | 是 | 0.987113 |
| 设备79 | Sparse | PatchTST | iTransformer | 是 | 0.991719 |
| 设备80 | Sparse | PatchTST | iTransformer | 是 | 0.990886 |
| 设备81 | Sparse | PatchTST | PatchTST | 否 | 0.998799 |
| 设备82 | Sparse | PatchTST | PatchTST | 否 | 0.986770 |
| 设备83 | Sparse | PatchTST | Informer | 是 | 0.976480 |
| 设备85 | Sparse | PatchTST | PatchTST | 否 | 0.977313 |
| 设备86 | Intermittent | iTransformer | PatchTST | 是 | 0.930052 |

## 输出与复现

- device_model_assignments.csv：全设备12维特征、模型、上轮模型及变化标记。
- cluster_summary.csv：簇级汇总。
- cluster_centers.csv：12维原始特征空间的聚类中心。
- run_metadata.json：标准化参数、随机种子、软件版本及路由定义。

检查：80台设备各一份文件，四级数据有限且非负；所有设备参与聚类并恰好分配一个模型；设备数合计80。
