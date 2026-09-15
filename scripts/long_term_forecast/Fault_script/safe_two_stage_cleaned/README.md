# SAFE 回归预训练 → 冻结主干 → 分类头训练

独立入口：`run_safe_two_stage.py`。默认仅检查数据，显式 `--train` 后才会执行训练。原联合训练入口 `run_safe_binary.py` 及其脚本仍可使用。

## 两阶段逻辑

1. **回归阶段**：用历史四级计数预测下一小时四个级别的原始计数，只有主干进入优化器，分类头不参与前向或更新。默认用四通道平均 MSE 优化、验证 MSE 早停；同时记录总体与各级别 MSE、MAE。可用 `--regression-loss mae` 改为 MAE 优化及早停，两项指标仍同时报告。MSE单位为次数平方，MAE单位为次数；不是对0/1标签计算回归误差，也不对原始计数标准化。
2. **重新加载最佳主干**：从磁盘读取验证集最优的 `best_backbone.pt`，并检查设备、模型、窗口长度、架构、特征顺序、数据SHA256和切分协议。不会误用最后一个epoch或旧联合分类检查点。
3. **冻结主干训练分类头**：全部主干参数 `requires_grad=False`、清空旧梯度、保持 `eval()`，主干前向使用 `no_grad()`；Dropout关闭，BatchNorm统计不更新。优化器仅包含97参数的 `4→16→1` 分类头。每个epoch及最佳分类检查点重载后，对主干全部参数和buffer做SHA256比对，变化即报错。
4. **分类标签及损失**：下一小时四个告警级别只要有一个大于0，标签即为1；沿用训练负例/正例比例加权的BCEWithLogitsLoss，阈值默认0.5，不在测试集调参。

保留原预测骨干路径（Informer解码器及蒸馏等），训练回归后四通道输出应表达告警次数；分类头基于这四个预测值学习分类，不读取未来真实计数。冻结主干后，其误差不能由分类损失修正，这是本方案的能力边界。

两阶段共用清洗CSV、70%/10%/20%时间切分及连续小时窗口集合。跨清洗断点的窗口被过滤。两阶段分别在自身训练结束后评估测试集，不用回归测试指标选择主干或指导分类早停。

固定设备路由仍为27/69→iTransformer、58→Informer、83→PatchTST；该路由来自全量清洗数据，严格评估需另在训练段拟合。`--model` 可显式覆盖，用于三种模型的统一对比。

## 依赖与预检查

在 `Time-Series-Library` 根目录、已安装CUDA PyTorch的环境中执行。依赖与上一版一致：

```bash
python -m pip install -r scripts/long_term_forecast/Fault_script/safe_binary_cleaned/requirements.txt
export PYTHON_BIN="$(command -v python)"
# 路由分配的16组，默认不训练
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all.sh
# 三种模型×四台设备×四种长度，共48组，默认不训练
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_models.sh
```

四份清洗CSV需单独上传至 `dataset/fault_selected_cleaned/`；不由Git同步。可设置 `DATA_ROOT`、`OUTPUT_ROOT`、`PYTHON_BIN` 环境变量。

## 在服务器启动两阶段训练

每个实验先完成回归，再冻结该最佳主干训练分类头；脚本按实验顺序执行，失败即停止。

```bash
# 固定路由：16组实验，每组包含两个阶段
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all.sh --train --device cuda:0

# 三种模型完整对比：48组实验，每组包含两个阶段
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_models.sh --train --device cuda:0

# 单个设备、模型、历史长度
python run_safe_two_stage.py --device-id 58 --model Informer --seq-len 24 \
  --train --device cuda:0
```

已有第一阶段主干时，固定设备路由的16组分类实验使用各自的
`stage1_regression/best_backbone.pt`，不会重新执行回归阶段：

```bash
# 默认只预检查；显式传入 --train 才启动分类头训练
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_heads.sh
bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_heads.sh \
  --train --device cuda:0 --epochs 100 --patience 3
```

后台运行示例：

```bash
mkdir -p outputs/safe_two_stage_cleaned/logs
nohup bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_models.sh \
  --train --device cuda:0 \
  > outputs/safe_two_stage_cleaned/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

分类头批量后台运行：

```bash
mkdir -p outputs/safe_stage2_classification/logs
LOG="outputs/safe_stage2_classification/logs/all_heads_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/run_all_heads.sh \
  --train --device cuda:0 --epochs 100 --patience 3 \
  > "$LOG" 2>&1 &
echo "PID: $!"
echo "日志: $LOG"
tail -f "$LOG"
```

`run_all_heads.sh`默认使用`outputs/safe_stage2_classification`作为独立输出根目录，
可通过`BACKBONE_ROOT`、`DATA_ROOT`、`OUTPUT_ROOT`和`PYTHON_BIN`覆盖路径或解释器。

默认回归学习率 `--learning-rate 1e-4`，分类头学习率 `--head-learning-rate 1e-3`；均使用原type1衰减。回归最多 `--regression-epochs 100`，分类头最多 `--epochs 100`；分别用验证损失早停，`--patience 3`。

## 分开执行两个阶段

```bash
# 只预训练回归主干（同样输出MSE和MAE）
python run_safe_two_stage.py --device-id 58 --seq-len 24 --stage regression \
  --train --device cuda:0

# 之后只训练头：模型、长度和数据必须与checkpoint匹配
python run_safe_two_stage.py --device-id 58 --seq-len 24 --stage head \
  --backbone-checkpoint /path/to/stage1_regression/best_backbone.pt \
  --train --device cuda:0
```

若预训练用了 `--model` 或非默认架构参数，head阶段必须传相同参数。`--backbone-checkpoint`仅用于head阶段。已有回归checkpoint可复用，但本入口仅接受本两阶段脚本生成且元数据匹配的checkpoint；不自动猜测其他格式。

如需分别比较MSE与MAE作为优化目标，执行两次并改变 `--regression-loss`；输出带时间戳，互不覆盖。固定一个主干时不需要为了报告MAE再训练一次。

## 独立输出

默认目录：

```text
outputs/safe_two_stage_cleaned/device58_Informer_L24/时间戳/
├── config.json
├── summary.json
├── stage1_regression/
│   ├── best_backbone.pt
│   ├── history.csv
│   ├── metrics.json
│   └── predictions.csv
└── stage2_head/
    ├── best_head.pt
    ├── history.csv
    ├── metrics.json
    ├── predictions.csv
    └── freeze_audit.json
```

`stage1_regression/history.csv`记录每轮train/val MSE、MAE，`metrics.json`记录最佳主干测试集整体及各级MSE、MAE；按全部窗口和四通道计算平均，而非等权平均批次。

`stage2_head/best_head.pt`包含完整冻结主干和分类头参数、配置和源主干校验值。推理重建后应调用 `freeze_backbone()`、`eval()`；冻结标志不是state_dict参数。`freeze_audit.json`记录训练前后主干参数及buffer完全相同的校验值。

汇总已完成实验的回归及分类指标：

```bash
python scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/collect_results.py
```

生成 `outputs/safe_two_stage_cleaned/comparison.csv`，每次运行保留一行，不自动按测试指标挑选最优实验。head-only运行没有本轮回归测试指标，汇总中相应列留空。

## 不启动训练的验证

```bash
python -m unittest discover -s tests -p 'test_safe*.py' -v
```

测试仅使用人工数据做梯度、冻结buffer、指标数值及检查点兼容性验证，不调用训练循环或optimizer.step。本版没有CUDA实测或收敛结论。Informer的ProbAttention在eval下仍随机采样，冻结表示参数及buffer不变，不保证每次预测完全相同；保留该原模型行为。
