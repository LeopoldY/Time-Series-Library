# 设备27、28、39、46、58：下一小时故障与故障类型联合预测

入口：`run_safe_fault_types.py`；批量脚本：`scripts/tools/train_fault_types.sh`。
交付目录未修改。该入口独立于旧二分类、两阶段回归训练；旧4通道检查点无法直接用于新增类型通道，需重新训练。

## 标签与训练数据

- 从原始 `设备编号_NetworkFaultLog.csv` 的 `名称` 字段解析具体类型，去除首尾空白；`告警类型`（普通/根源等）不是具体类型，不作标签。沿用旧任务口径：四个级别的告警均属于预测事件；这里的“故障”指日志告警发生，不代表独立确认的物理故障。
- 不按修复时长、严重程度、类型频次删告警。同一ID重复只保留一次；同ID的名称/级别/来源/发生时间冲突直接报错。不同ID即使名称和时间一致也保留。缺名称、时间、级别或设备不匹配报错。
- 每台设备独立建立整点左闭右开小时网格，首尾不完整小时排除。保留全部中间零值小时，不再删除336小时长零段。假设两端之间日志采集持续可用；没有独立采集状态，无法区分真实安静与采集中断。若有停采记录，需另行引入观测掩码。
- 以完整小时行数按70%/10%/20%连续时间划分训练/验证/测试。输入过去L个已结束小时，标签为紧接的下一小时发生的类型集合；不是当前小时识别，也不是持续故障状态预测。边界可以使用此前已观测历史，目标小时互斥。
- 训练时段建立每台设备的类型词表。每小时输入为四级计数+各训练类型计数，先log1p，均值和标准差仅从训练小时拟合。新类型发生后，其级别计数可进入之后的历史输入，但不会事后扩展词表。
- 同一小时允许多个类型同时为1。二分类标签为下一小时任意告警是否发生。未来清除时间、确认时间、Repair_Interval、ID均不作为输入。
- 完整数据中的类型目录只用于审计和测试指标；测试首次出现的类型保留在真值中，模型对应预测为0，不隐藏无法学习的类型。输出明确统计未见类型与测试事件数。

## 模型与评估

Informer / iTransformer / PatchTST 的预测分支接收动态通道数，预测表示与历史末小时/均值拼接，经联合头输出1个二分类logit和K个类型logit。端到端同时更新主干和分类头，不冻结旧回归主干。

损失=二分类加权BCE+各类型加权BCE均值；正例权重仅由训练目标计算，负正比限制在[1,50]。默认64维主干、4头、两层、AdamW、验证联合损失早停。最佳checkpoint重载后，阈值仅用验证集按每个输出F1在0.05至0.95间选择，同分取高阈值；验证集单一类别时固定0.5。测试集只在模型和阈值固定后评估。

最终先判断是否故障，再输出过阈值的多个类型。判断无故障时类型集合为空；判断有故障但没有可靠类型时输出`__UNKNOWN_TYPE__`（有告警但类型未确定，不代表模型已识别新类型）。训练目标无正例的类型禁止输出。此保守规则使系统始终返回一致结果，也意味着稀有/新类型不能保证被识别。

指标包含二分类Precision/Recall/F1/AP和混淆矩阵，各类型支持数/Precision/Recall/F1/AP，类型micro-F1、测试有正例类型macro-F1、全目录macro-F1和完整集合准确率。无正例类型AP为null，其F1按0记录；macro-F1的口径分开保存。`__UNKNOWN_TYPE__`单独统计为未定类型告警，不作为一个真实类型参与micro-F1；完整集合准确率将它判为不匹配。另有全正常基线和上一小时类型集合持久性基线。概率分数未经校准。

## 服务器启动

在已有CUDA PyTorch环境中执行。要求PyTorch≥2.2，NumPy<2；原仓库旧requirements中的torch 1.7.1不适用于安全检查点重载。已有环境满足依赖时无需重装；否则按服务器CUDA配置安装PyTorch后安装其余依赖（见`requirements-fault-types.txt`）。

```bash
cd /你的服务器路径/Time-Series-Library
git switch main
git pull --ff-only origin main
python -c "import torch,pandas,sklearn,reformer_pytorch; print(torch.__version__, torch.cuda.is_available())"
```

原始数据默认在`dataset/fault_raw/raw_data`；不能传只有四级计数的旧清洗文件。如存放在其他位置，用`--raw-root /实际原始CSV目录`。数据不随Git上传，服务器需要这五台设备的原始CSV。

先做全部180组数据预检查，不启动训练：

```bash
bash scripts/tools/train_fault_types.sh \
  --raw-root dataset/fault_raw/raw_data \
  --output-root outputs/fault_types_preflight
```

确认审计、类型支持数和划分后后台启动：

```bash
mkdir -p outputs/safe_fault_types/logs
LOG="outputs/safe_fault_types/logs/train_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/tools/train_fault_types.sh \
  --raw-root dataset/fault_raw/raw_data \
  --output-root outputs/safe_fault_types \
  --train --device cuda:0 \
  > "$LOG" 2>&1 &
echo "PID: $!  LOG: $LOG"
tail -f "$LOG"
```

默认串行运行5设备×3模型×4历史长度(3/6/12/24)×3种子(2021/2022/2023)，最多100epoch、patience=10、batch=64；避免同时占用一张卡。`CUDA_VISIBLE_DEVICES=1`时仍使用`--device cuda:0`。显存不足可传`--batch-size 16`。

先运行一组确认服务器环境：

```bash
bash scripts/tools/train_fault_types.sh \
  --devices 27 --models iTransformer --lengths 24 --seeds 2021 \
  --epochs 3 --train --device cuda:0
```

本地验证实验的复现设置：

```bash
python run_safe_fault_types.py --devices 27 28 39 46 58 \
  --models iTransformer --lengths 24 --seeds 2021 \
  --epochs 20 --patience 5 --d-model 32 --device cpu --train \
  --output-root outputs/fault_types_local_validation
```

此版本不支持epoch级断点续训，也不自动跳过旧任务。每次新建时间戳目录；失败即停止，`completion.json`保持false。部分完成后可用`--devices`、`--models`、`--lengths`、`--seeds`选择剩余组合新建批次；不要把多个批次重复窗口合并统计。

## 输出与验收

```text
outputs/safe_fault_types/时间戳/
  completion.json                 # 全部任务成功才complete=true；training区分预检查
  prepared/设备编号/hourly.csv     # 清洗后的小时计数；保留完整类型目录用于审计
  prepared/设备编号/audit.json     # 源SHA256、清洗数量、训练词表、归一化、未见类型
  summary.csv                     # 每个实验的测试指标和持久性基线
  seed_summary.csv                # 同设备/模型/L跨种子的均值和样本标准差
  device编号_模型_L长度_seed种子/
    config.json
    data_audit.json                # 各分割目标数与各类型正例数
    best.pt                       # 模型权重、配置、词表、归一化、支持掩码和验证阈值
    history.csv
    thresholds.json
    metrics.json                  # 完整逐类型指标和基线
    predictions.csv               # 目标小时、故障真值/预测、类型集合和类型分数
```

预检查只生成数据与审计，不生成权重、测试指标或种子汇总。单种子汇总std为空。模型输入仅选训练词表，不将hourly.csv中的测试新增类型列直接喂入模型。

实验比较应看逐类型召回、macro-F1和基线，不能仅看整体准确率。当前本地实验用于验证完整训练与评估链路，不能代替服务器180组比较，也不能用测试集指标选阈值或重训选优。

测试命令：`python -m pytest tests/test_fault_types.py -q`。
