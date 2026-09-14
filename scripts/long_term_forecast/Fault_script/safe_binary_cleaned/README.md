# SAFE 清洗数据下一小时故障二分类

## 已落地的适配

- 独立入口 `run_safe_binary.py`，不走旧的 `run.py` / `Exp_Fault_Classification`，不修改原回归代码。
- 输入 `dataset/fault_selected_cleaned/设备{27,58,69,83}_cleaned.csv`，四列按 `w_level1,w_level2,w_level3,w_level4` 固定顺序读取，不标准化、不改变原始计数。
- 标签为下一小时任一级别大于0，输出单个0/1标签。仅用连续小时窗口，检查历史窗口及目标之间所有间隔，跳过跨清洗断点的窗口。
- 时间顺序70%/10%/20%切分，标签必须在对应区间；验证/测试可以使用前一区间的已知历史。
- 固定上轮清洗四设备路由：27→iTransformer，58→Informer，69→iTransformer，83→PatchTST。路由来自全量清洗数据，仅适合既定设备画像实验；严格无未来信息评估应另行在训练段重算路由。本实现按已确认分配固定执行，不隐式重聚类。
- 原骨干保持 `long_term_forecast` 分支、原编码/解码/归一化计算。输入/输出通道设为4，单步输出后增加 `Linear(4,16) → GELU → Dropout(0.1) → Linear(16,1)`，共97个分类头参数。Informer原输出投影由1通道配置为4通道；这是为头读取四级表示所需的维数配置调整。
- 骨干和头联合优化 BCEWithLogitsLoss，`pos_weight=训练负窗口数/训练正窗口数`；不使用固定的1000倍权重。原预测输出经联合分类训练后不再承诺是次数预测。
- 保留历史长度3/6/12/24、原路由的编码器层数、batch size、factor和PatchTST分块规则，默认d_model=512、n_heads=8、d_ff=2048、Adam学习率1e-4、type1学习率调整、验证BCE早停。
- 固定分类阈值0.5（可提前配置），不基于测试结果调阈值；只有训练结束重载最佳检查点后评估一次测试集。本版使用float32，不提供AMP或多卡并行路径。

## 上传服务器

建议上传完整 `Time-Series-Library`，包括新入口、`models/SAFE_Binary.py`、`data_provider/safe_binary_data.py`、本脚本目录、原来的models/layers/utils依赖，以及四个清洗CSV。所有默认路径由脚本自身位置推导，无本机绝对路径。

在服务器现有可用PyTorch环境中安装本目录requirements；新环境先安装适合服务器CUDA的PyTorch，再安装其他依赖。该入口无需安装旧项目requirements中的全部模型依赖，不建议用其torch==1.7.1覆盖新环境。

```bash
cd /your/server/path/Time-Series-Library
python3 -m pip install -r scripts/long_term_forecast/Fault_script/safe_binary_cleaned/requirements.txt
```

## 仅检查数据（默认，无训练）

```bash
bash scripts/long_term_forecast/Fault_script/safe_binary_cleaned/run_all.sh
```

输出16种设备/历史长度组合的实际窗口及正负例数量；不构造模型、不创建训练输出目录。单设备可执行 `device27.sh` 等脚本。

## 稍后在服务器显式启动训练

以下命令仅供服务器执行，本次适配未执行：

```bash
bash scripts/long_term_forecast/Fault_script/safe_binary_cleaned/run_all.sh --train --device cuda:0
# 仅设备58的四种历史长度
bash scripts/long_term_forecast/Fault_script/safe_binary_cleaned/device58.sh --train --device cuda:0
# 仅一个实验
python3 run_safe_binary.py --device-id 58 --seq-len 24 --train --device cuda:0
```

批量脚本顺序执行16个实验，任一失败即停止。`CUDA_VISIBLE_DEVICES=1` 时应使用逻辑设备 `--device cuda:0`。CPU需显式传 `--device cpu`；CUDA不可用会报错，不会静默转CPU。

可通过环境变量 `PYTHON_BIN`、`DATA_ROOT`、`OUTPUT_ROOT` 指定解释器、数据和输出目录，例如：

```bash
OUTPUT_ROOT=/data/experiments/safe_binary_cleaned \
  bash scripts/long_term_forecast/Fault_script/safe_binary_cleaned/run_all.sh --train --device cuda:0
```

## 独立输出

默认 `outputs/safe_binary_cleaned/device{ID}_{Model}_L{长度}/{时间戳}/`，每次启动生成新目录，不覆盖原回归结果：

- `config.json`：超参、输入SHA256、标签规则、特征顺序、窗口统计、类别权重、路由口径。
- `best.pt`：骨干和分类头的最佳参数、配置、最佳epoch与验证损失。
- `history.csv`：逐epoch训练/验证BCE和学习率。
- `metrics.json`：Precision、Recall、F1、AP、ROC-AUC、混淆矩阵、始终正常与上一小时标签基线。
- `predictions.csv`：设备、下一小时真实时间、标签、Sigmoid分数、分类判断及阈值。

无正例时正类Recall/F1/AP记null；单类时ROC-AUC记null。类别加权后的Sigmoid是分类分数，不能直接视为已校准故障概率。训练窗口缺少任一类别则明确报错。

本版从头联合训练，不自动加载旧回归检查点，不提供中断续训；检查点可用于同配置重建模型与推理。

## 无训练验证

```bash
python3 -m unittest discover -s tests -p 'test_safe_binary.py' -v
```

测试仅进行人工序列标签/断点/切分验证、三个原骨干前向与梯度检查、检查点往返和指标边界检查；不调用训练入口或optimizer.step，不产生训练结果。
