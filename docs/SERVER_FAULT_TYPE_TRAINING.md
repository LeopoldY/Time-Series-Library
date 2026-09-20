# 五设备故障类型预测：原模型路由、两阶段、L=6/12，共10组

入口：`run_safe_fault_types.py`；脚本：`scripts/tools/train_fault_types.sh`。适配PyTorch **2.1.0**，包括对应CUDA构建；NumPy<2。交付目录不变。

## 固定实验协议

沿用原串行两阶段入口 `scripts/tools/run_routed_two_stage_serial.py` 默认读取的分配表：`docs/safe_all_selected_cleaned_clustering/device_model_assignments.csv`。不重新聚类、不遍历三个模型；这里只沿用设备与模型的映射，不复用旧四通道清洗数据。

| 设备 | 原分配模型 | 历史长度L |
| --- | --- | --- |
| 27 | Informer | 6、12 |
| 28 | iTransformer | 6、12 |
| 39 | Informer | 6、12 |
| 46 | PatchTST | 6、12 |
| 58 | Informer | 6、12 |

固定单种子2021，5设备×2长度=**10组实验**，每组包含两个顺序执行的阶段。一个设备/长度完成回归、分类与测试后，才开始下一组。`plan.json`保存实际任务、路由来源和SHA256；缺失、重复或非法模型路由直接报错。其他历史分配表不混用。

1. **回归预训练**：输入过去L小时的四级计数和训练词表中的类型计数，主干预测下一小时的全部4+K个通道。回归目标采用与输入一致的log1p及训练段标准化；MSE损失，仅优化主干。按验证MSE早停，保存并重载最佳 `best_backbone.pt`。
2. **故障发生及类型预测**：冻结上述最佳主干的参数，清空其梯度，保持eval模式及BatchNorm统计不变，主干前向使用no_grad。仅训练分类头，输出1个二分类logit和K个类型logit。主干预测表示与历史末小时/均值拼接后进入分类头。每轮及最佳分类检查点重载后校验主干全部参数与buffer的哈希。

第二阶段损失=二分类加权BCE+各类型加权BCE均值；权重只由训练目标计算，负正比限制在[1,50]。回归和分类各自按验证损失选择最佳检查点；分类阈值仅由验证集F1选择（0.05至0.95，单一类别固定0.5），之后才评估测试集。两阶段均用Adam和原type1学习率衰减；回归初始学习率0.0001，分类0.001。默认每阶段最多100epoch，patience=10，batch=64，主干64维/4头/两层。

保留“先回归，再冻结主干做预测”的训练协议；回归通道扩展到类型计数，不能直接加载原四通道回归检查点。此前端到端故障类型检查点也不适用于当前两阶段流程。

## 原始日志、清洗与标签

- 输入 `设备编号_NetworkFaultLog.csv`，具体类型来自`名称`，不是表示普通/根源告警属性的`告警类型`。四个级别都纳入预测；“故障”指日志告警发生，不直接等同于独立确认的物理故障。
- 去除名称首尾空白，同ID去重；同ID关键字段冲突、必需字段缺失、设备来源不匹配或未知级别时报错。不同ID事件保留，不按类型频次、修复时长删事件。
- 整点左闭右开小时网格；首尾不完整小时排除。保留所有中间零值小时，不删除336小时长零段。假设观测区间内采集持续可用；没有采集状态记录，无法区分安静时段与采集中断。
- 按完整小时行数70%/10%/20%划分训练/验证/测试，目标小时互斥。输入t-L至t-1，目标t；跨分割边界允许使用此前已观测历史。
- 每台设备的类型词表仅由训练段拟合；输入为4+K个计数，经log1p及训练段均值/标准差归一化。回归未来目标单独返回，绝不作为模型历史输入。清除时间、确认时间和修复时长均不作输入。
- 下一小时可以同时存在多个类型，因此使用多标签Sigmoid。判断无故障时输出空集合；判断有故障却无类型过阈值时输出`__UNKNOWN_TYPE__`，表示类型未确定。训练目标无正例的类型禁止输出。
- 测试新增类型保留在真值和完整目录中，其模型预测为0，不用测试数据扩展训练词表。指标包括二分类混淆矩阵/F1/AP、逐类型支持数/Precision/Recall/F1/AP、类型micro/macro-F1和完整集合准确率，并比较全正常及上一小时持久性基线。
- 类型micro-F1只评价具体类型；未知输出另行统计，完整集合准确率将其判为不匹配。分别报告测试有正例类型及全目录的macro-F1；无正例类型AP为null、F1为0。

## 服务器命令

进入仓库并激活现有PyTorch 2.1.0 CUDA环境，无需升级至2.2。原仓库旧requirements的torch 1.7.1不适用于本入口，依赖见`requirements-fault-types.txt`。

```bash
cd /你的服务器路径/Time-Series-Library
git switch main
git pull --ff-only origin main
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

原始日志默认位于`dataset/fault_raw/raw_data`，可用`--raw-root`指定其他目录；不能传只有四级计数的旧清洗数据。数据不随Git上传。

预检查10组，不启动训练：

```bash
bash scripts/tools/train_fault_types.sh --output-root outputs/fault_types_preflight
```

后台启动全部10组，每组自动依次执行两个阶段：

```bash
mkdir -p outputs/safe_fault_types/logs
LOG="outputs/safe_fault_types/logs/two_stage_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/tools/train_fault_types.sh \
  --raw-root dataset/fault_raw/raw_data \
  --train --device cuda:0 > "$LOG" 2>&1 &
echo "PID: $!  LOG: $LOG"
tail -f "$LOG"
```

默认就是设备27/28/39/46/58、L=6/12、seed=2021，不需额外指定。`--models`和`--seeds`已移除，避免误启动模型或种子的笛卡尔积；单一种子通过`--seed`指定。可用`--devices 27 --lengths 6`选择其中一组；长度仅允许6或12。`--regression-epochs`控制第一阶段上限，`--epochs`控制第二阶段上限。显存不足可用`--batch-size 16`。

每次新建时间戳目录；无epoch续训或自动跳过旧任务。失败即停止，completion记录失败任务及已完成数；解决问题后可选择剩余设备/长度新建批次。不可将重复时间窗口合并统计。所有任务成功且 `training=true, complete=true, expected=10, completed=10` 才代表默认训练批次完成，单纯预检查的training为false。

## 输出

```text
outputs/safe_fault_types/时间戳/
  plan.json
  completion.json
  summary.csv
  seed_summary.csv                 # 单种子均值等于单次结果，std为空
  prepared/设备编号/hourly.csv
  prepared/设备编号/audit.json
  device编号_模型_L长度_seed2021/
    config.json
    data_audit.json
    stage1_regression/
      best_backbone.pt
      history.csv
      metrics.json                 # 标准化log1p空间的测试MSE/MAE
    stage2_head/
      best_head.pt                 # 完整权重、词表、归一化、阈值及主干来源
      history.csv
      freeze_audit.json
      thresholds.json
      metrics.json
      predictions.csv
```

不启动训练的兼容性检查：

```bash
python -m pytest tests/test_fault_types_torch_compat.py -q
```

该检查仅含前向推理、序列化、路由验证，以及用替身替换epoch执行的阶段控制测试，不执行反向传播或优化器更新。代码适配验证不等同于服务器真实两阶段训练已完成。
