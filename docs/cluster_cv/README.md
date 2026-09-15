# 按聚类合并设备训练：3模型 × 4历史长度 × 10折

新入口：`run_safe_cluster_cv.py`；批量脚本：`scripts/tools/train_cluster_cv.sh`。

## 数据和训练方式

- 使用上一版 `dataset/fault_paper2016/four_levels` 的87台清洗数据，清洗与十折协议不变。
- 每一折，只用各设备当前训练段计算12维画像并拟合标准化器及KMeans；按簇平均稀疏程度映射 Dense→Informer、Intermittent→iTransformer、Sparse→PatchTST。每折独立路由，设备成员可能随折变化，不使用测试数据固定簇。
- 把同簇设备的小时行合并为**一份带device_id的数据集**，每折保存3份CSV。保持设备独立序列，按设备内部划分及构造窗口，再合并样本送入同一个模型；不将设备计数相加，device_id只用于追溯，不参与特征。
- 保留3/6/12/24小时历史、下一小时四级计数回归和任一级告警二分类。每个模型/长度/折只训练一套共享参数，合计120个实验，不再每设备训练一个模型。
- 每台设备先切十个时间块：第k块测试，第(k+1)%10块验证，其余训练。历史与目标必须同属一个集合，且时间连续；不跨设备、不跨断点、不跨训练/验证/测试边界。离线十折仍可能使用测试时段之后的数据训练，不等同于滚动回测。
- 训练集按窗口等权采样；数据多的设备贡献更多梯度。不增加设备等权采样或将其计数混为网络总量。
- 单台设备某集合无窗口时，在逐设备审计中记录0，其他设备仍可训练；单台设备单一标签不剔除。只有整个簇缺少训练/验证/测试窗口或整个簇训练集单一类别时，预检报错，不静默少跑120组。
- 每组从头初始化模型和Adam；使用簇训练集的负/正样本比例作为BCE正类权重，验证联合损失早停。重新加载最佳模型后仅评估一次测试集。

## 全量预检

87台设备、10折、3模型、4长度全部通过，**120/120组可训练**。各组样本数见 `preflight.csv`；逐折成员见 `routing_summary.csv`。预检不构建模型、不更新权重。

## 服务器启动

在仓库根目录，使用已安装CUDA PyTorch的训练环境：

```bash
git pull --ff-only origin main
python -m pip install -r scripts/long_term_forecast/Fault_script/safe_binary_cleaned/requirements.txt
# 若已有上一轮清洗结果，可省略这一行
python scripts/tools/prepare_paper2016.py

mkdir -p outputs/safe_cluster_cv
export RUN_DIR="outputs/safe_cluster_cv/run_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/tools/train_cluster_cv.sh --device cuda:0 \
  > "${RUN_DIR}.log" 2>&1 &
echo "PID=$! RUN_DIR=$RUN_DIR"
```

查看进度：`tail -f "${RUN_DIR}.log"`。默认串行120组，最多100轮/组，patience=5，batch_size=32，主干d_model=512。可传 `--epochs`、`--patience`、`--batch-size` 等参数。

仅预检：

```bash
python run_safe_cluster_cv.py --run-dir outputs/safe_cluster_cv/preflight --device cuda:0
# 同一参数加 --train，可直接从该计划开始训练
```

中断后，将 `RUN_DIR` 设置为原目录，使用**相同参数**再次运行启动脚本。输入文件SHA256、参数及聚合数据校验通过后，已完成任务直接跳过；未完成任务从头训练并保留旧尝试目录。完成标志仅在预测、指标和模型全部写出后生成，不把半完成实验纳入均值。

`DATA_ROOT`和`PYTHON_BIN`环境变量可指定数据路径与解释器；`--folds 0`或`--lengths 3`仅用于调试，不是完整120组实验。

## 输出及十折平均报告

```text
RUN_DIR/
  plan.json                       # 全120组计划、输入哈希、参数、逐设备样本审计
  preflight.csv
  fold0..fold9/
    routing/                      # 训练段拟合的聚类与路由
    datasets/{模型}.csv           # 每折每簇一份聚合小时数据
  jobs/{折_模型_长度}/
    complete.json                 # 已完成实验、输出路径与哈希
    devicecluster_.../时间戳/     # 模型、训练曲线、预测、指标
  fold_metrics.csv                # 120行逐折测试指标
  cv_summary.csv                  # 12行：每模型×每长度的十折均值/样本标准差
  REPORT.md                       # 可读报告
  status.json                     # 完成数及是否为完整120组
```

报告按每折整个簇的全部测试窗口先计算指标，再对十折**等权平均**（不是按折样本量加权；不是每设备指标的平均）。报告MSE、MAE、Accuracy、Precision、Recall、F1、Balanced Accuracy、FPR、AP及ROC-AUC。每项保留有效折数；缺折或某折该指标未定义时，不以较少折数冒充十折平均，均值留空。三模型服务于不同簇，指标不能当作同一测试集上的模型排名。

每完成一组自动刷新报告；也可单独汇总：

```bash
python scripts/tools/summarize_cluster_cv.py "$RUN_DIR"
```

本地仅做预检及小架构冒烟验证，正式十折平均数需服务器完成训练后由上述报告给出。

## 验证范围

- 既有18项SAFE测试通过；新增4项测试覆盖设备/集合/时间断点隔离、单类设备保留、测试数据不影响聚类、十折均值和缺折规则。
- 6台合成设备、3模型×4长度×10折，共120组小架构CPU训练全部完成；再次启动跳过全部120组，修改参数后拒绝混用旧结果。
- 真实87台设备的全120组预检通过；另用fold0、L3、小架构各训练一个聚合模型，验证真实数据训练、检查点重载和预测导出。
- 预测文件按device_id/target_time核对身份、样本数与唯一性，并由预测值重算MSE/MAE；详情见 `validation.json`。这些是流程验证，不是正式收敛结果。
