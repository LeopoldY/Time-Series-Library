# 清洗后全设备两阶段串行训练实施方案

日期：2026-09-15。适用入口：`run_safe_two_stage.py`；批量串行入口：`scripts/tools/run_routed_two_stage_serial.py`。本方案配套代码已完成预检，尚未启动真实设备训练。

## 一、实施目标与不可变协议

使用 `dataset/fault_all_selected_cleaned` 的87台设备数据，每台设备仅训练本轮路由指定的目标模型。每个“设备、历史长度、随机种子”组合严格按以下顺序运行：

**路由模型回归训练 → 选择并重新加载最佳回归检查点 → 冻结主干参数和缓冲区 → 训练history分类头 → 验证集选择检查点及阈值 → 分类测试评估。**

当前任务完成后才启动下一个任务，不并行启动多个训练进程，不将回归与分类损失联合反向传播，不在第二阶段微调主干。这里的串行是“每个组合先回归后分类，再进入下一组合”，并非要求87台全部回归结束后才能启动第一台分类。

预测时点为下一小时t，历史输入为 `[t-L,t)`；四级原始告警计数为唯一观测输入。二分类真值固定为：

`y[t] = 1{max(w_level1[t], w_level2[t], w_level3[t], w_level4[t]) > 0}`。

任一级、任意正数量均为有故障；四级全为零才为无故障。分类头只输出一个logit。ASL的负样本截断和最终分数阈值均不改变此标签定义。

## 二、数据、路由与任务清单

### 2.1 数据版本

- 目录：`dataset/fault_all_selected_cleaned`，文件格式：`设备N_cleaned.csv`。
- 列顺序：`date,w_level1,w_level2,w_level3,w_level4`；每行是1小时窗口，date记录窗口中点。
- 共87个文件、536750行，149761条告警；清洗删除了350998个连续长全零小时，告警总数守恒。
- 清洗口径：统一小时窗口后，删除四级同时为零且连续至少336小时的完整区间；保留其他计数和原时间戳。本方案不再次清洗、不重采样、不跨断点补数据。
- SHA256依据：`docs/safe_all_selected_cleaned_clustering/device_model_assignments.csv` 中的 `Input_SHA256`；批量入口执行前核查路由集合与数据集合完全一致、设备无重复、文件哈希一致，每个任务启动前再次核查数据哈希。
- 设备58当前数据相较旧四设备小时表保留了15条额外原始告警，不能直接假定旧58检查点可复用。

仓库既有 `.gitignore` 排除 `dataset/` 与 `outputs/`；本次发布实施方案、代码、路由与审计清单，训练数据需预先放到服务器上述目录。原始事件数据存在时可运行 `python3 scripts/tools/clean_all_like_selected.py` 生成对应数据与路由；软件版本变化可能改变聚类结果，复现本轮训练应优先使用发布的固定路由及匹配的数据哈希。

### 2.2 固定路由

以 `docs/safe_all_selected_cleaned_clustering/device_model_assignments.csv` 为唯一设备到模型映射，不使用旧四设备默认路由。

| 类别 | 目标模型 | 设备数 | 设备编号 |
| --- | --- | ---: | --- |
| Dense | Informer | 4 | 27、36、39、58 |
| Intermittent | iTransformer | 3 | 28、42、86 |
| Sparse | PatchTST | 80 | 0—86范围内除上述7台之外的设备 |

每条训练命令必须显式传 `--model` 与 `--data-root`。例如本轮设备27使用Informer、69使用PatchTST；旧入口的默认映射不适用于本轮全设备路由。

**评估边界：**本轮路由使用全历史数据聚类，属于固定全历史设备画像条件下的实验。即使回归与分类严格按时间切分，也不能将其称作完全无未来信息的在线模型选择评估。正式前瞻评估需另建仅训练段拟合的路由并重跑；该调整不属于本轮固定路由主实验。

### 2.3 时间划分与有效窗口

每台设备保留n行清洗数据：`train_end=floor(0.7n)`，`val_end=n-floor(0.2n)`。按目标行索引划分训练 `[0,train_end)`、验证 `[train_end,val_end)`、测试 `[val_end,n)`。

历史长度L的目标t必须满足t≥L，且从t-L至t的所有相邻时间差均为1小时，否则剔除窗口。验证/测试历史可以使用预测时点之前已观测的数据；任何未来目标值都不进入历史。该切分按清洗后行数而非原始日历时长计算。

不对四级计数做全数据标准化、不修改目标尺度。保留各主干代码内部已有的归一化机制，分类头内部仅对计数特征作log1p变换。

### 2.4 实验规模与单类例外

| 阶段 | 历史长度 | 种子 | 回归任务数 | 分类任务数 |
| --- | --- | --- | ---: | ---: |
| 首轮可运行性与结果基线 | 24 | 2021 | 87 | 85 |
| 主实验：独立输出四种长度 | 3、6、12、24 | 2021 | 348 | 340 |
| 完整三种子重复（包含主实验） | 3、6、12、24 | 2021、2022、2023 | 1044 | 1020 |

设备59、62在四种长度的训练段均无正类：保留回归训练，分类状态记录 `blocked_single_class_train`，不使用伪正类、不从验证/测试借正样本、不将其标记为分类训练完成。后续需要补充合法历史样本或单独设计跨设备训练方案，另立实验后再恢复分类。

全量预检348个组合均有训练、验证、测试有效窗口。L=24训练段合计33342正样本、335353负样本，负正比约10.06:1；各设备正类比例中位数约3.72%。详见 `docs/SAFE_imbalance_head/class_distribution.csv`。

## 三、超参数总表

以下是可直接执行的第一版固定配置，值与当前训练入口一致；未声称经过数据调优后最优。除明确的长度、种子重复外，不在测试集上选择配置。

### 3.1 数据和主干架构

| 参数 | Informer | iTransformer | PatchTST | 设计说明 |
| --- | --- | --- | --- | --- |
| L / seq_len | 主实验3、6、12、24 | 同左 | 同左 | 首轮先用24小时；所有长度只预测下一小时 |
| pred_len | 1 | 1 | 1 | 固定预测跨度 |
| enc_in / dec_in / c_out | 4/4/4 | 4/4/4 | 4/4/4 | 统一四级接口；不需要decoder的主干忽略对应输入 |
| d_model | 512 | 512 | 512 | 沿用当前接口默认表征维度 |
| n_heads | 8 | 8 | 8 | 每头维度64 |
| d_ff | 2048 | 2048 | 2048 | 前馈层为d_model的4倍 |
| e_layers | 3 | 2 | 2 | 由configure按模型设置 |
| d_layers | 1 | 不适用 | 不适用 | Informer解码层 |
| dropout | 0.1 | 0.1 | 0.1 | 主干原有设置 |
| 激活 | GELU | GELU | GELU | 原有主干设置 |
| factor | 3 | 接口保留 | 接口保留 | Informer ProbAttention采样因子 |
| distil | True | 不适用 | 不适用 | Informer保留蒸馏结构 |
| label_len | 1 | 接口保留 | 接口保留 | decoder为最后已知计数+下一步零占位，不输入真实未来计数 |
| embed/freq | timeF/h | timeF/h | 接口保留 | 使用已知日历信息，具体消费方式由主干决定 |
| batch_size | 32 | 32 | 16 | 两个阶段使用相同batch设置 |

PatchTST随L设置：

| L | patch_len | stride | 右侧padding |
| ---: | ---: | ---: | ---: |
| 3 | 1 | 1 | 1 |
| 6 | 3 | 1 | 1 |
| 12 | 6 | 3 | 3 |
| 24 | 12 | 6 | 6 |

表内e_layers、d_layers、factor、主干dropout、patch_len、stride由现有configure自动设置，不是当前命令行可以独立覆盖的选项。改变这些规则需修改代码、记录新协议，不能只在方案中写一个不会生效的参数。

### 3.2 第一阶段：目标模型回归训练

| 参数 | 设定 | 理由/执行含义 |
| --- | --- | --- |
| 任务 | long_term_forecast，下一小时四级计数 | 保留路由主干的预测结构 |
| 优化器 | Adam，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 | 与当前实现默认值一致 |
| 初始学习率 | 1e-4 | 对应 `--learning-rate` |
| 损失 | MSE，对B×4元素取均值 | 主实验固定；MAE作为独立对照，不与MSE实验混选检查点 |
| 最大epoch | 100 | 早停优先于上限 |
| patience | 3 | 连续3轮验证目标未严格改善则停止 |
| 最佳检查点 | 最低验证MSE | 保存 `stage1_regression/best_backbone.pt`，不是最后一轮权重 |
| 数据采样 | 训练shuffle=True；验证/测试False | 不改变时间切分或窗口内部顺序 |
| drop_last / workers | False / 0 | 保留最后不足batch的样本、简化复现 |
| 精度 / 梯度裁剪 | float32 / 不启用 | 当前实现无AMP及裁剪步骤 |

MSE可能更偏重大计数误差，不能据此保证分类更优。若做MAE对照，应使用独立输出目录、相同数据划分与主干配置，并用对应验证MAE选检查点，之后完整执行第二阶段。

### 3.3 第二阶段：冻结主干的二分类训练

| 参数 | 设定 | 理由/执行含义 |
| --- | --- | --- |
| 主干加载 | 重新加载最佳回归检查点 | 核验设备、模型、L、架构、四级顺序、数据哈希、协议 |
| 冻结范围 | 主干所有参数和缓冲区 | requires_grad=False且eval；包括BatchNorm状态，优化器只接收head参数 |
| head_type | history | 4维预测分支+20维历史统计分支，融合后单logit |
| head_hidden / head_dropout | 16 / 0.1 | 新分类头985参数 |
| 输出偏置 | log(N_positive/N_negative) | 仅使用训练有效目标窗口统计 |
| 优化器 | Adam，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 | 新建优化器，不继承回归动量 |
| 初始学习率 | 1e-3 | 对应 `--head-learning-rate` |
| 损失 | asymmetric | sigmoid ASL，批内样本均值 |
| gamma_neg / gamma_pos / clip | 4 / 0 / 0.05 | 抑制易负样本，保留正类交叉熵梯度 |
| pos_weight / 过采样 | 不使用 / 不使用 | 避免与ASL重复补偿 |
| 最大epoch / patience | 100 / 3 | 两阶段共享patience=3 |
| 检查点选择 | 验证AP最大 | 同AP取较早epoch；验证集单类时回退验证loss最小 |
| 决策阈值 | 验证F1最大，beta=1 | 只用验证集搜索不同分数；并列取更高阈值 |
| 单类验证阈值 | 固定0.5 | 标记回退原因，不使用测试标签调阈值 |

ASL：`p=sigmoid(z)`、`q=max(p-0.05,0)`，`loss=-y*log(p)-(1-y)*q^4*log(1-q)`。输出仍是“下一小时任意一级故障”的单一分类分数；clip不是标签阈值，也不是推理阈值。F2仅作为事先定义的偏召回对照（`--threshold-beta 2`），不根据测试表现切换。

历史分支每级使用：最近计数、窗口均值、峰值、非零比例、归一化距最近故障步数；前三项log1p，预测分支使用signed-log1p。全部只取预测时点之前的信息。详细结构和论文依据见 [不平衡分类头设计](SAFE_imbalance_head/README.md)。

### 3.4 学习率调度的精确定义

两阶段均沿用当前 `fit_stage` 的type1衰减。第e轮实际使用：

`lr(e) = lr_initial × 0.5^max(e-2, 0)`。

因此第1、2轮均为初始值，第3轮减半，第4轮为1/4，第10轮为1/256。回归前四轮为 `1e-4,1e-4,5e-5,2.5e-5`；分类为 `1e-3,1e-3,5e-4,2.5e-4`。这不是恒定学习率，也不是“从第二轮即减半”。

该衰减较快，最大100轮并不意味着长时间有效优化。首轮应检查训练/验证曲线；若表现为过早停止学习，可另建恒定/余弦调度对照，但当前入口尚无对应调度参数，主实验不得静默更改实现。

## 四、串行执行步骤与命令

所有命令在仓库根目录 `Time-Series-Library` 执行。环境需要匹配项目的PyTorch、NumPy、pandas、scikit-learn、einops、reformer-pytorch等依赖；先跑测试，再核查GPU。现有实现支持cpu/cuda:N，未支持MPS。固定软件版本、设备型号及git提交号到实验记录；随机种子不保证跨硬件完全逐位一致。

### 步骤1：实现测试与全量预检

```bash
python3 -m unittest discover -s tests -p 'test_safe*.py'

python3 scripts/tools/run_routed_two_stage_serial.py \
  --lengths 3 6 12 24 --seeds 2021 --device cuda:0 \
  --plan-path outputs/safe_all_serial/plan.json
```

不带 `--execute` 只检查数据/路由/有效窗口并写计划，不构建主干、不训练。预期87台、348个回归任务、340个分类任务；任何文件缺失、路由错配、哈希变化或无有效窗口，均在全量预检时失败，尚未启动任何训练。

### 步骤2：单设备确认完整两阶段

以设备27的路由Informer、L=24为例：

```bash
python3 run_safe_two_stage.py \
  --device-id 27 --model Informer --seq-len 24 \
  --data-root dataset/fault_all_selected_cleaned \
  --output-root outputs/safe_serial_pilot \
  --stage both --regression-loss mse \
  --regression-epochs 100 --epochs 100 --patience 3 \
  --learning-rate 1e-4 --head-learning-rate 1e-3 \
  --batch-size 32 --num-workers 0 --seed 2021 \
  --d-model 512 --n-heads 8 --d-ff 2048 \
  --head-type history --head-hidden 16 --head-dropout 0.1 \
  --head-loss asymmetric --asl-gamma-neg 4 --asl-gamma-pos 0 --asl-clip 0.05 \
  --head-selection ap --threshold-policy val_fbeta --threshold-beta 1 --threshold 0.5 \
  --device cuda:0 --train
```

检查回归检查点、冻结审计、分类decision及CSV一致性。该pilot结果只验证流程，不用其测试成绩重新选择全设备超参数；其输出与正式主实验隔离。

### 步骤3：运行全量串行主实验

```bash
python3 scripts/tools/run_routed_two_stage_serial.py \
  --lengths 3 6 12 24 --seeds 2021 --device cuda:0 \
  --output-root outputs/safe_all_serial_main \
  --plan-path outputs/safe_all_serial_main/plan.json \
  --execute
```

入口内每条命令显式设置第三节参数，使用 `subprocess.run` 等待当前组合完成后再启动下一组合；分类可训练时传 `--stage both`，设备59、62等单类组合传 `--stage regression` 并保留分类阻塞状态。任一实际训练任务非零退出即停止后续任务，不静默吞掉失败。

如先做L=24首轮，指定 `--lengths 24` 和新的输出目录。三种子完整重复可指定 `--seeds 2021 2022 2023`；若已做2021主实验，只在另一目录补2022、2023，避免重复计数。主干及分类头每个种子都重新训练，不选择“测试上最好”的种子。

### 步骤4：失败恢复或仅重训分类头

批量入口刻意不提供自动跳过成功任务的隐式续跑：`output_root/execution` 已存在时拒绝覆盖。每任务有独立 `.log` 与状态 `.json`；先定位失败原因。中断后仍为running的任务视作未完成，不当作成功。

若回归阶段已成功，使用其最佳回归检查点单独启动分类：

```bash
python3 run_safe_two_stage.py \
  --device-id 27 --model Informer --seq-len 24 \
  --data-root dataset/fault_all_selected_cleaned \
  --output-root outputs/safe_serial_recovery \
  --stage head --backbone-checkpoint /absolute/path/to/stage1_regression/best_backbone.pt \
  --regression-loss mse --seed 2021 --batch-size 32 --num-workers 0 \
  --d-model 512 --n-heads 8 --d-ff 2048 \
  --epochs 100 --patience 3 --head-learning-rate 1e-3 \
  --head-type history --head-hidden 16 --head-dropout 0.1 \
  --head-loss asymmetric --asl-gamma-neg 4 --asl-gamma-pos 0 --asl-clip 0.05 \
  --head-selection ap --threshold-policy val_fbeta --threshold-beta 1 --threshold 0.5 \
  --device cuda:0 --train
```

设备、模型、L、架构、特征顺序、数据哈希及协议必须匹配；模型、数据或架构不匹配时必须重跑回归，不能绕过校验。恢复会重新初始化分类头及优化器，不是继续上次分类epoch；由于随机数消耗顺序不同，不保证与一次性 `both` 运行逐位相同。

后续未启动组合可从plan中取出options，用 `run_safe_two_stage.py`加上数据/输出路径和`--train`按序执行，记录新的输出目录；不得直接删除旧实验目录再伪装成无中断实验。资源不足需降低batch时，建立有记录的新配置并对两个阶段统一使用，当前批量入口不接受batch覆盖，需显式单任务命令或版本化修改。

## 五、检查点、输出与验收

每个组合独立保存：

```text
outputs/<experiment>/execution/
  deviceN_Model_L24_seed2021.json       # 运行状态及完整命令
  deviceN_Model_L24_seed2021.log
  deviceN_Model_L24_seed2021/
    deviceN_Model_L24/<timestamp>/
      config.json
      summary.json
      stage1_regression/
        history.csv
        best_backbone.pt
        metrics.json
        predictions.csv
      stage2_head/
        history.csv
        best_head.pt
        freeze_audit.json
        validation_scores.csv         # val_fbeta模式
        decision.json
        metrics.json
        predictions.csv
```

### 5.1 必须通过的验收项

1. 路由表、计划、运行config中的设备/模型对应关系完全一致；有效窗口数与预检一致。
2. 回归最佳epoch仅由验证MSE决定；分类检查点仅由验证AP决定，单类验证回退有据可查。
3. `freeze_audit.json`中训练前后主干参数/缓冲区哈希相同，分类优化器范围为head only。
4. 真实标签逐行等于目标小时四级任意非零，预测输入与目标间不存在时间断点。
5. `decision.json`的阈值和检查点中的decision一致；分类CSV回读满足 `y_pred=(probability>=threshold)`。
6. 无NaN/Inf；不存在重复设备任务或静默漏跑。340个可训练分类任务须完整输出，8个阻塞组合单列，不计作成功分类实验。

### 5.2 指标与选择规则

回归报告总体与逐级MSE、MAE；分类重点报告正类AP、Precision、Recall、F1、FPR、TP/FP/FN/TN，Accuracy和ROC-AUC作为补充；保留始终无故障和上一小时标签持久性基线。AP使用average_precision定义，不与梯形积分PR-AUC混称。

每台设备分别报告L=3/6/12/24四组结果，不把重叠时间戳的多种长度预测拼接成独立样本。若需确定唯一部署长度，应按预先规定的验证AP选择（并列选较短L）；不得按测试F1选择。当前代码保存逐运行结果，尚未自动实施跨长度部署选择，需另汇总验证指标并记录选择表。

各设备和各长度的测试正类比例可能不同，先逐设备比较，再作设备宏平均，另报微平均；单类指标标记未定义，不伪填0。多种子报告均值和标准差，不择优汇报单个种子。由于设备59、62阻塞，必须同时报告可训练覆盖率85/87。

当前 `both` 实现在确定最佳回归权重后立即导出回归测试结果，然后启动分类阶段；该结果不进入分类训练、检查点或阈值选择。执行期间不得人工根据这些测试输出调整后续超参数。最终性能改进需真实训练结果支持，不能由本方案或合成测试推出。

## 六、实施里程碑与交付

| 里程碑 | 交付 | 通过条件 |
| --- | --- | --- |
| M0 数据与计划核验 | 路由哈希、全部数据哈希、任务计划、类分布 | 覆盖87设备且识别全部单类阻塞组合 |
| M1 单设备串行联调 | 一个路由主干的两阶段全套文件 | 最佳回归检查点加载成功、冻结审计和标签检查通过 |
| M2 全量主实验 | 348回归结果、340分类结果、8项阻塞记录 | 无未解释失败、无模型路由错误 |
| M3 重复与消融 | 多种子结果、旧MLP/BCE与新head/ASL对照 | 同数据/主干/划分，模型与阈值选择均基于验证集 |
| M4 发布实验报告 | 逐设备指标、覆盖率、配置、版本、检查点索引 | 训练事实与方案参数对应，明确固定全历史路由的评估边界 |

GPU时长和显存没有通过512维主干的真实全量训练测量，本方案不编造耗时；M1记录每epoch耗时、峰值显存、实际早停轮数，再据各模型设备数估算总时长。当前串行流程主干在分类阶段仍逐batch前向计算，尚未缓存预测，因此分类阶段并非只运行985参数的小网络。

## 七、本次实施准备结果

- 已核验3/6/12/24全部348个组合的路由、数据哈希和有效窗口，分类可训练数为340；摘要见 `safe_serial_training_preflight.json`。
- 配套不平衡分类头已通过15项原有及新增测试，三个主干的合成两阶段流程、仅分类恢复入口均验证通过；新增批量计划器测试及本次复核另记录于预检摘要。
- 本次工作交付方案及执行工具，不自动启动真实训练。
- 相关文件：[不平衡分类头设计](SAFE_imbalance_head/README.md)、[清洗与路由说明](safe_all_selected_cleaned_clustering/README.md)、[实现验证](SAFE_imbalance_head/validation.md)。
