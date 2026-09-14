# SAFE 全设备聚类与模型分配

日期：2026-09-14。直接调用当前 `cluster.py` 的 `extract_features_from_file` 与 `perform_clustering`，从 `SAFE.py` 读取模型映射。未修改算法或原始数据，未启动训练。

## 范围与方法

扫描 `dataset/fault`，发现 80 份 CSV，有效设备 80 台，跳过 0 份。按设备编号升序固定输入顺序。

沿用当前代码的全量时间段统计：Sparsity = w_level4 为 0 的窗口比例；Mean_Intensity = w_level4 均值；CV = 总体标准差 / 均值（均值不大于 1e-6 时取 0）。三项特征经 StandardScaler 标准化，执行 KMeans(n_clusters=3, random_state=42, n_init=10)，按各簇平均稀疏度升序命名 Dense、Intermittent、Sparse。

本次是全量数据画像与模型分配；后续若用于严格预测评估，应仅用训练段重新拟合，避免未来数据参与模型选择。Dense 等类别是当前样本集合内的相对类别，不表示绝对故障频繁；零四级告警设备仍按原规则参与聚类。

## 聚类汇总

| 簇编号 | 类别 | 模型 | 设备数 | 平均稀疏度 | 平均强度 | 平均 CV |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Dense | Informer | 1 | 0.918267 | 0.228146 | 5.039531 |
| 2 | Intermittent | iTransformer | 31 | 0.986748 | 0.017646 | 10.063514 |
| 0 | Sparse | PatchTST | 48 | 0.997434 | 0.003083 | 22.438726 |

## 全部设备分配

| 设备 | 设备类别 | 聚类 | 模型 | 稀疏度 | 均值 | CV | 四级正例窗口数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 设备0 | S5300_S5328C-EI | Intermittent | iTransformer | 0.979714 | 0.046060 | 8.640827 | 207 |
| 设备1 | S5300_S5328C-EI | Sparse | PatchTST | 0.998138 | 0.001862 | 23.152810 | 19 |
| 设备2 | S5300_S5328C-EI | Intermittent | iTransformer | 0.988926 | 0.011368 | 9.566371 | 113 |
| 设备3 | S5300_S5328C-EI | Intermittent | iTransformer | 0.991866 | 0.008134 | 11.042634 | 83 |
| 设备4 | S9300_S9303 | Sparse | PatchTST | 0.996864 | 0.003234 | 18.081821 | 32 |
| 设备5 | S9300_S9303 | Sparse | PatchTST | 0.997256 | 0.002744 | 19.063803 | 28 |
| 设备6 | S9300_S9303 | Sparse | PatchTST | 0.997648 | 0.002352 | 20.595307 | 24 |
| 设备7 | S5300_S5328C-EI | Sparse | PatchTST | 0.994806 | 0.009604 | 28.954351 | 53 |
| 设备8 | S5300_S5328C-EI | Sparse | PatchTST | 0.995590 | 0.012642 | 17.603535 | 45 |
| 设备9 | S5300_S5328C-EI | Sparse | PatchTST | 0.996178 | 0.008624 | 17.754990 | 39 |
| 设备10 | S5300_S5328C-EI | Sparse | PatchTST | 0.997648 | 0.002352 | 20.595307 | 24 |
| 设备11 | S5300_S5328C-EI | Sparse | PatchTST | 0.997648 | 0.002352 | 20.595307 | 24 |
| 设备12 | S9300_S9303 | Intermittent | iTransformer | 0.994512 | 0.005586 | 13.575672 | 56 |
| 设备13 | S9300_S9303 | Intermittent | iTransformer | 0.994512 | 0.005488 | 13.461586 | 56 |
| 设备14 | S5300_S5328C-EI | Intermittent | iTransformer | 0.977460 | 0.047334 | 7.506184 | 230 |
| 设备15 | S9300_S9303 | Sparse | PatchTST | 0.998922 | 0.001078 | 30.440702 | 11 |
| 设备16 | S5300_S5328C-EI | Sparse | PatchTST | 0.997452 | 0.002548 | 19.785387 | 26 |
| 设备17 | S5300_S5328C-EI | Sparse | PatchTST | 0.996374 | 0.003626 | 16.576604 | 37 |
| 设备18 | S5300_S5328C-EI | Sparse | PatchTST | 0.996178 | 0.003822 | 16.144381 | 39 |
| 设备19 | S5300_S5328C-EI | Intermittent | iTransformer | 0.986476 | 0.018718 | 9.857852 | 138 |
| 设备20 | S5300_S5328C-EI | Sparse | PatchTST | 0.996178 | 0.006664 | 18.645806 | 39 |
| 设备21 | S5300_S5328C-EI | Sparse | PatchTST | 0.997746 | 0.002254 | 21.039301 | 23 |
| 设备23 | S5300_S5328C-EI | Intermittent | iTransformer | 0.993826 | 0.006174 | 12.687327 | 63 |
| 设备24 | S5300_S5328C-EI | Intermittent | iTransformer | 0.993728 | 0.006272 | 12.587196 | 64 |
| 设备25 | S5300_S5328C-EI | Intermittent | iTransformer | 0.993728 | 0.006370 | 12.681282 | 64 |
| 设备26 | S9300_S9306 | Sparse | PatchTST | 0.998138 | 0.001862 | 23.152810 | 19 |
| 设备27 | S9300_S9306 | Sparse | PatchTST | 0.997942 | 0.002058 | 22.020553 | 21 |
| 设备28 | S9300_S9306 | Sparse | PatchTST | 0.998040 | 0.001960 | 22.565460 | 20 |
| 设备29 | S9300_S9303 | Sparse | PatchTST | 0.997942 | 0.002156 | 22.471837 | 21 |
| 设备30 | S9300_S9303 | Sparse | PatchTST | 0.997942 | 0.002058 | 22.020553 | 21 |
| 设备31 | S5300_S5328C-EI | Sparse | PatchTST | 0.998138 | 0.001862 | 23.152810 | 19 |
| 设备32 | S9300_S9303 | Sparse | PatchTST | 0.997942 | 0.002058 | 22.020553 | 21 |
| 设备33 | S5300_S5328C-EI | Sparse | PatchTST | 0.997256 | 0.002744 | 19.063803 | 28 |
| 设备34 | S5300_S5328C-EI | Sparse | PatchTST | 0.997648 | 0.002450 | 20.971714 | 24 |
| 设备35 | S5300_S5328C-EI | Sparse | PatchTST | 0.997746 | 0.002254 | 21.039301 | 23 |
| 设备36 | S9300_S9303 | Sparse | PatchTST | 0.997550 | 0.002450 | 20.178206 | 25 |
| 设备37 | S5300_S5328C-EI | Sparse | PatchTST | 0.997256 | 0.002842 | 19.368232 | 28 |
| 设备38 | S5300_S5328C-EI | Sparse | PatchTST | 0.996276 | 0.005194 | 19.873460 | 38 |
| 设备39 | S9300_S9303 | Intermittent | iTransformer | 0.991474 | 0.016170 | 14.098573 | 87 |
| 设备40 | S5300_S5352C-EI | Sparse | PatchTST | 0.996864 | 0.003136 | 17.829049 | 32 |
| 设备42 | S9300_S9303 | Intermittent | iTransformer | 0.993924 | 0.006174 | 12.888371 | 62 |
| 设备43 | S9300_S9303 | Sparse | PatchTST | 0.997060 | 0.002940 | 18.415573 | 30 |
| 设备44 | S5300_S5328C-EI | Intermittent | iTransformer | 0.983242 | 0.016758 | 7.659799 | 171 |
| 设备45 | S5300_S5328C-EI | Intermittent | iTransformer | 0.983242 | 0.016758 | 7.659799 | 171 |
| 设备46 | S9300_S9303 | Sparse | PatchTST | 0.996178 | 0.003822 | 16.144381 | 39 |
| 设备47 | S5300_S5328C-EI | Sparse | PatchTST | 0.997746 | 0.002254 | 21.039301 | 23 |
| 设备48 | S5300_S5328C-EI | Sparse | PatchTST | 0.997746 | 0.002254 | 21.039301 | 23 |
| 设备49 | S5300_S5328C-EI | Intermittent | iTransformer | 0.984418 | 0.020188 | 9.481163 | 159 |
| 设备50 | S9300_S9303 | Intermittent | iTransformer | 0.987848 | 0.014994 | 10.071577 | 124 |
| 设备51 | S9300_S9303 | Intermittent | iTransformer | 0.987848 | 0.014994 | 10.114765 | 124 |
| 设备52 | NE20E-8 | Intermittent | iTransformer | 0.973834 | 0.046452 | 8.894646 | 267 |
| 设备53 | NE20E-8 | Intermittent | iTransformer | 0.980204 | 0.024108 | 7.682935 | 202 |
| 设备54 | S5300_S5328C-EI | Intermittent | iTransformer | 0.980890 | 0.020188 | 7.369188 | 195 |
| 设备55 | S5300_S5328C-EI | Intermittent | iTransformer | 0.980792 | 0.020482 | 7.404799 | 196 |
| 设备56 | S9300_S9303 | Sparse | PatchTST | 0.999216 | 0.000784 | 35.700140 | 8 |
| 设备57 | S5300_S5328C-EI | Intermittent | iTransformer | 0.988926 | 0.011270 | 9.529780 | 113 |
| 设备58 | S5300_S5328C-EI | Sparse | PatchTST | 0.999510 | 0.000490 | 45.164145 | 5 |
| 设备60 | S9300_S9306 | Sparse | PatchTST | 0.999314 | 0.000686 | 38.166927 | 7 |
| 设备61 | S9300_S9306 | Sparse | PatchTST | 0.999314 | 0.000686 | 38.166927 | 7 |
| 设备63 | S5300_S5328C-EI | Sparse | PatchTST | 0.996080 | 0.004018 | 16.125087 | 40 |
| 设备64 | S5300_S5328C-EI | Sparse | PatchTST | 0.995982 | 0.004214 | 16.074108 | 41 |
| 设备65 | S5300_S5328C-EI | Sparse | PatchTST | 0.996276 | 0.004018 | 16.861175 | 38 |
| 设备66 | S5300_S5328C-EI | Sparse | PatchTST | 0.996374 | 0.003920 | 17.098684 | 37 |
| 设备67 | S5300_S5328C-EI | Sparse | PatchTST | 0.996668 | 0.003724 | 19.143068 | 34 |
| 设备69 | S9300_S9303 | Sparse | PatchTST | 0.998726 | 0.001568 | 33.392552 | 13 |
| 设备70 | S9300_S9303 | Sparse | PatchTST | 0.998726 | 0.001568 | 33.392552 | 13 |
| 设备71 | S5300_S5328C-EI | Intermittent | iTransformer | 0.986476 | 0.016758 | 9.491001 | 138 |
| 设备72 | S5300_S5328C-EI | Intermittent | iTransformer | 0.986182 | 0.017150 | 9.378818 | 141 |
| 设备73 | S9300_S9303 | Intermittent | iTransformer | 0.986672 | 0.016268 | 9.294671 | 136 |
| 设备74 | S9300_S9303 | Intermittent | iTransformer | 0.986574 | 0.016562 | 9.263731 | 137 |
| 设备75 | NE20E-8 | Intermittent | iTransformer | 0.978832 | 0.038612 | 9.196918 | 216 |
| 设备76 | NE20E-8 | Intermittent | iTransformer | 0.981478 | 0.022540 | 7.915456 | 189 |
| 设备78 | S5300_S5352C-EI | Intermittent | iTransformer | 0.993630 | 0.006468 | 12.581379 | 65 |
| 设备79 | S5300_S5328C-EI | Intermittent | iTransformer | 0.993140 | 0.007154 | 12.258452 | 70 |
| 设备80 | S5300_S5328C-EI | Intermittent | iTransformer | 0.984810 | 0.015484 | 8.126178 | 155 |
| 设备81 | S9300_S9303 | Sparse | PatchTST | 0.998040 | 0.002156 | 23.391211 | 20 |
| 设备82 | S5300_S5328C-EI | Sparse | PatchTST | 0.996668 | 0.003724 | 18.001616 | 34 |
| 设备83 | NE40E-X3 | Dense | Informer | 0.918267 | 0.228146 | 5.039531 | 834 |
| 设备85 | S9300_S9306 | Sparse | PatchTST | 0.998040 | 0.002058 | 23.047373 | 20 |
| 设备86 | S9300_S9306 | Sparse | PatchTST | 0.997844 | 0.002254 | 21.936969 | 22 |

## 复核与文件

- `device_model_assignments.csv`：全设备特征、类别、模型、数据路径、窗口数与输入文件校验值。
- `cluster_summary.csv`：各簇数量及原始特征空间均值。
- `run_metadata.json`：运行参数、依赖版本及源代码校验值。

已检查设备无重复、目标列数值有限且非负、每个有效设备均分配模型。固定输入顺序及软件版本有助于复现；不同输入顺序或依赖版本可能影响 KMeans 结果，簇编号本身没有业务含义。
