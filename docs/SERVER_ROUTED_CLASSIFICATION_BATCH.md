# 按聚类路由完整执行分类阶段并汇总

新增入口：`run_safe_routed_batch.py`。使用已完成回归训练的主干，为路由表中的每台设备、每种历史长度自动训练冻结主干上的分类头，并评估测试集。无需重新跑回归。

## 范围与输入

默认读取 `docs/safe_cleaned_four_devices_clustering/device_model_assignments.csv`：设备27/69→iTransformer，58→Informer，83→PatchTST；长度3、6、12、24，共16组。设备集合来自CSV，不再由批量程序写死。

如果“所有设备”指原始80台，必须另行提供80台对应的清洗CSV和回归主干，并通过 `--routes` 指定对应路由表。默认四设备表不能代表80台设备已经验证。清洗数据文件统一命名为 `设备编号_cleaned.csv`，由 `--data-root` 指定目录；无需复制路由CSV中记录的本机绝对路径。

每组主干来自：

```text
回归输出根目录/device编号_模型_L长度/时间戳/stage1_regression/best_backbone.pt
```

相邻实验根目录应保留原 `config.json`。脚本读取真实检查点配置与验证分数，不依赖文件名猜测模型。

## 行为

1. 读取路由表中的设备及模型，检查重复、非法模型及数据文件；若路由表有Input_SHA256，校验清洗数据一致性。
2. 匹配设备、模型、长度、预训练损失、数据SHA256和数据协议一致的主干。默认只选择MSE预训练；MAE需传 `--regression-loss mae`。
3. 多个候选时选择回归验证误差最低的主干；同分按路径排序。不会按测试指标筛选，也不会直接取最新时间戳。
4. 任一组合缺失则整体报错，尚不启动任何训练。主干架构的最终严格加载检查仍由每组子入口在更新参数前完成。
5. 保存固定计划 `plan.json`，记录主干及数据文件SHA256、具体路径、候选数与验证分数。
6. 显式 `--execute` 才训练。各组都传入 `--stage head`，读取自己的最佳主干，冻结主干参数和BatchNorm统计，仅训练97参数分类头。
7. 每组成功后从 `predictions.csv` 重新统计指标，不通过目录扫描任意选一个历史分类结果。部分失败会输出缺失清单，绝不报告整批完成。

输入/目标、断点处理、BCE类别权重、阈值规则沿用已有两阶段实现。该入口会重新训练分类头，不是加载已训练分类头做纯推理；只有回归主干被复用。

## 服务器使用

进入仓库并激活原PyTorch环境：

```bash
cd /你的服务器路径/Time-Series-Library
git pull --ff-only origin main
```

先生成计划（读取检查点并写计划文件，不训练）：

```bash
python run_safe_routed_batch.py \
  --checkpoint-root outputs/safe_two_stage_cleaned \
  --data-root dataset/fault_selected_cleaned \
  --output-root outputs/safe_routed_classification \
  --device cuda:0
```

记下程序打印的 `Batch:` 目录，检查该目录的 `plan.json`。它应包含16组任务。若数据或主干缺失，先根据报错补齐；程序不会使用随机初始化主干代替。

执行已保存计划：

```bash
BATCH="$(pwd)/outputs/safe_routed_classification/实际批次时间戳"
nohup python -u run_safe_routed_batch.py \
  --resume "$BATCH" --execute \
  > "$BATCH/batch.log" 2>&1 &
echo "PID: $!"
tail -f "$BATCH/batch.log"
```

也可以一步生成计划并执行：

```bash
python -u run_safe_routed_batch.py \
  --checkpoint-root outputs/safe_two_stage_cleaned \
  --data-root dataset/fault_selected_cleaned \
  --execute --device cuda:0
```

建议先生成计划，便于核对主干来源。`--device`、`--epochs`、`--patience`、`--batch-size`、`--head-learning-rate`、`--threshold` 在生成计划时保存；resume沿用计划里的参数，忽略重新传入的这些设置。需要改变超参时创建新计划。

## 中断与继续

```bash
python -u run_safe_routed_batch.py --resume "$BATCH" --execute
```

成功完成的任务跳过；未完成或失败的任务使用新的尝试目录，从该回归主干重新训练分类头。不是从分类头中断epoch续训。已有输出不覆盖；同一批次有进程锁，禁止同时重复执行。

继续前会检查计划中所有主干和数据文件的SHA256；源文件被覆盖后会报错，应恢复原文件或重新生成计划。

## 输出与汇总

```text
outputs/safe_routed_classification/批次时间戳/
├── plan.json
├── state.json
├── completion.json
├── all_metrics.csv
├── by_length.csv
├── REPORT.md
└── jobs/设备模型长度/尝试时间戳/
    ├── train.log
    └── 设备模型长度/分类运行时间戳/
        ├── config.json
        ├── summary.json
        └── stage2_head/
            ├── best_head.pt
            ├── predictions.csv
            ├── metrics.json
            ├── history.csv
            └── freeze_audit.json
```

- `all_metrics.csv`：每设备、每长度的Accuracy、Precision、Recall、F1、AP、ROC-AUC、FPR、混淆矩阵、样本数、阈值、主干路径和分类运行路径。
- `by_length.csv`：同一长度下全部设备的宏平均及混淆矩阵微平均；只有该长度全部设备完成后才生成对应行。不会将不同长度的重复时间窗口混在一起计算一个总分。
- `REPORT.md`：全部设备、不同长度结果表与缺失任务。
- `completion.json`：必须 `complete=true` 且 `completed=expected` 才表示整批完成。程序成功结束也会打印 `Complete`。
- AP采用average precision；无正类等未定义指标留空。预测文件中的阈值及标签必须一致，否则拒绝汇总。

单独重新汇总已有批次，不训练：

```bash
python run_safe_routed_batch.py --summarize "$BATCH"
```

不完整批次仍会写出已完成部分和缺失清单，并以非零退出码结束。

## 更多设备

```bash
python run_safe_routed_batch.py \
  --routes /你的路由表.csv \
  --data-root /所有设备的清洗CSV目录 \
  --checkpoint-root /所有设备的回归输出目录
```

路由表至少有 `Device_Name,Model`；设备名格式为 `设备27`，支持Informer/iTransformer/PatchTST。可有 `Label,Input_SHA256`。原80设备聚类结果来自未清洗数据，其分配不等同于80台清洗数据重新聚类；应使用与你的实验数据口径一致的路由表。

## 验证

本次不运行服务器训练。本地人工检查覆盖：验证指标选主干、缺失组合拒绝启动、参数继承、head-only命令构造、从预测文件重算指标和不完整批次状态；原冻结主干及分类测试继续运行。服务器真实主干不在本机，完整16组执行需在服务器完成。
