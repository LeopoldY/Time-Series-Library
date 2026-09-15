# 面向无故障多数类的二分类头设计

## 1. 任务协议与问题

标签严格保持：`y[t] = 1{任意 k∈{1,2,3,4}, w_level_k[t] > 0}`；模型仅接收 `[t-L,t)` 历史，预测下一小时 `t`。无论故障级别、告警数量，只要任一级非零就为正类。输出一个logit，经sigmoid和决策阈值生成0/1，不设置“至少两级故障”或“连续多个窗口故障”等新条件。

数据继续按时间70%/10%/20%划分训练、验证、测试；历史可使用预测时点之前的观测，但滑窗及目标边不能跨清洗断点。`data_provider/safe_binary_data.py`的标签与划分代码未修改。

对本轮87台清洗设备重新按有效滑窗统计：L=24训练段有33342个正样本、335353个负样本，负正比约10.06:1；每台设备正类比例的中位数约3.72%。这与“每级零值比例”不同，分类应使用四级取OR之后的标签分布。设备59、62的训练段没有正样本，现有训练入口会明确拒绝为它们训练二分类头；不能靠损失函数创造缺失的正样本。完整L=3/6/12/24和三段统计见 `class_distribution.csv`。

旧头为 `4→16→1` MLP（97参数），只看到回归预测，配合 `pos_weight=N_negative/N_positive` 的BCE。潜在限制是回归均值压缩了故障发生信息，且统一正类权重不能区分易负样本与难负样本。历史结果中设备58的排序与召回不一致，设备69则排序能力本身也偏弱；这些是设计动机，不构成新方案有效的实验证据。

## 2. 论文依据与采用范围

| 文献 | 已有思想 | 本项目采用部分 |
| --- | --- | --- |
| Kang et al., ICLR 2020, [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217) | 将表征学习与分类器训练分开 | 沿用先回归预训练、再冻结主干只训练分类头；本项目未声称复现论文的cRT采样方案 |
| Lin et al., ICCV 2017, [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | 减小容易分类样本的损失贡献 | 作为聚焦难样本的基础；默认使用下述非对称版本 |
| Ridnik et al., ICCV 2021, [Asymmetric Loss for Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) | 对正负样本采用不同聚焦强度并截断容易负样本 | 把sigmoid ASL应用到单个OR标签；不把任务改成四标签分类 |

ASL实现核对了[作者代码](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)。本实现采用样本均值、稳定log-sigmoid及完整聚焦权重梯度；与作者某个默认实现的求和、停止聚焦权重梯度选项有所区别。历史特征、双分支融合和残差连接是针对本项目的工程设计，不是上述论文中已经验证的时序架构。

## 3. 新分类头

```text
历史计数 X[t-L:t] ── 冻结的已训练预测主干 ── 4维下一小时预测 ── Linear(4,h)+GELU ─┐
      └───────────── 20维历史统计 ─────────────────── Linear(20,h)+GELU ───────┤
                                                                            ↓
                                                    拼接 → Linear(2h,h)+GELU+Dropout
                                                                            ↓
                        拼接的24维原始特征 → Linear(24,1,无bias) ──── 相加 ← Linear(h,1)
                                                                            ↓
                                                               单logit → sigmoid分数
```

每级历史统计5项，共20维：

1. 最近一次观测计数 `log1p(x[t-1])`。
2. 历史窗口平均计数 `log1p(mean(x))`。
3. 历史峰值 `log1p(max(x))`。
4. 窗口内该级非零小时比例。
5. 距最近一次该级非零观测的步数/L；窗口内从未出现则为1，最近一小时出现则为0。这是窗口内截断的间隔，不推断窗口之前的事件。

主干的4维预测使用 `sign(f)*log1p(abs(f))`，保留回归可能产生的负值；历史原始计数仍非负。对数压缩减轻大计数的尺度影响，不使用未来统计量。默认h=16、dropout=0.1，参数量985。所有新增可训练参数均在 `model.head` 下，冻结审计继续覆盖主干参数及BatchNorm等缓冲区。

输出偏置初始化为训练有效滑窗的 `log(N_positive/N_negative)`；其他权重保留常规随机初始化，因此不是每条样本初始分数都精确等于先验。

## 4. 不平衡损失

设模型分数 `p=sigmoid(z)`，负类使用 `p_minus=max(p-m,0)`：

`L = -y*(1-p)^gamma_pos*log(p) -(1-y)*p_minus^gamma_neg*log(1-p_minus)`。

默认 `gamma_pos=0, gamma_neg=4, m=0.05`：正类保留完整交叉熵学习信号；低分且正确的负类大幅减权，其中p≤0.05的负类贡献为零。m只影响训练损失，不是最终分类阈值，也不改变标签。

默认不与 `pos_weight`、过采样或额外先验logit校正叠加。过度补偿可能增加误报，故保留独立的BCE基线供消融。ASL输出与加权BCE输出都应视为分类分数，不能直接解释为校准后的实际故障概率。默认超参数来自既有方法及本任务的设计取舍，尚未经本数据验证集搜索证明最优。

## 5. 模型选择与阈值

推荐用验证AP（average precision）选择分类头epoch，避免用多数类主导的Accuracy决定检查点；若验证段只有单类，则显式回退到验证损失。回归主干仍按原MSE/MAE验证损失选择。

加载最佳分类头后，仅在验证集上精确枚举不同分数阈值，最大化F1；同分选择更高阈值以减少误报。若业务明确更重视漏报，可预先指定beta=2，改用F2。单类验证集不调阈值，回退到配置的固定值（默认0.5），在decision.json记录原因。

阈值与来源先写入 `decision.json` 和 `best_head.pt`，然后才生成分类测试预测；`validation_scores.csv`保留阈值选择证据。测试输出仍含 `probability` 列以兼容旧格式，其语义为未校准sigmoid分数。最终运行时须同时加载检查点中的模型配置、权重及 `decision.threshold`。

报告AP、Recall、Precision、F1、FPR和混淆矩阵，同时报告始终正常和上一小时标签持久性基线。不要仅凭Accuracy较高认定有效。验证AP选epoch与验证F1选阈值共用同一验证段，存在选择方差；需以未参与选择的测试段报告最终效果。

## 6. 使用方式

新方案是可选择的实现，旧命令默认仍为旧MLP、加权BCE、验证损失选epoch和0.5阈值，便于复现基线。推荐新方案参数：

```bash
--head-type history --head-loss asymmetric \
--head-selection ap --threshold-policy val_fbeta --threshold-beta 1
```

使用与当前数据SHA256、设备、主干架构及历史长度一致的已有回归检查点，仅重训分类头：

```bash
python3 run_safe_two_stage.py \
  --device-id 69 --model PatchTST --seq-len 24 \
  --data-root dataset/fault_all_selected_cleaned \
  --stage head --backbone-checkpoint /absolute/path/to/stage1_regression/best_backbone.pt \
  --head-type history --head-loss asymmetric \
  --head-selection ap --threshold-policy val_fbeta --threshold-beta 1 \
  --output-root outputs/safe_imbalance_head --device cuda:0 --train
```

例中设备69使用本轮87设备路由的PatchTST。旧四设备路由仍可能是iTransformer，不能将该权重当作PatchTST加载。新全设备数据中设备58多15条告警，旧58检查点的数据哈希不匹配时应重新训练对应主干，不能绕过兼容性检查。其他架构参数也须与原检查点一致。若没有兼容检查点，改用 `--stage both` 并移除 `--backbone-checkpoint`，正常运行两个阶段。

当前批量入口 `run_safe_routed_batch.py`继续使用旧固定阈值协议；新方案通过 `run_safe_two_stage.py`逐设备运行，避免批量汇总错误地将验证选择阈值当作固定0.5。

## 7. 消融与验收

在同一数据版本、主干检查点、随机种子、历史长度和时间划分下对比：

| 组别 | 分类头 | 损失 | 决策 |
| --- | --- | --- | --- |
| A 旧基线 | mlp | weighted_bce | fixed 0.5 / loss选epoch |
| B 决策改进 | mlp | weighted_bce | val_fbeta / AP选epoch |
| C 损失改进 | mlp | asymmetric | 同B |
| D 完整方案 | history | asymmetric | 同B |

另加history+weighted_bce可分离历史特征与损失的交互作用。仅用验证集选择超参数；至少3个预设随机种子报告逐设备AP/F1/Recall/FPR均值与波动，再按设备作宏平均。全局微平均不能替代稀疏设备的逐设备结果。若有明确误报预算，应在验证集约束FPR后再选择阈值，此版尚未加入该业务约束。

本次交付为论文依据、可运行实现、分布核查及协议/数值/集成测试，不宣称性能已提高。当前工作区没有已训练的回归检查点，因此不能直接给出在真实冻结主干上的新旧效果对比。验证记录见 `validation.md`。
