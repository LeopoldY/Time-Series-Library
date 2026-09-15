# 服务器第二阶段分类训练指南

更新日期：2026-09-15。适用于已经使用 `run_safe_two_stage.py --stage regression` 完成回归预训练的实验。

本阶段读取 `stage1_regression/best_backbone.pt`，冻结主干，仅训练 `4→16→1` 分类头。下一小时任一告警级别大于0即为正类。本文只提供操作指导，本次没有在本地或服务器启动训练。

## 1. 进入仓库，激活原训练环境

下文所有命令都在服务器 `Time-Series-Library` 根目录执行。将路径替换为实际仓库位置，并激活第一阶段使用的 Python 环境。

```bash
cd /你的服务器路径/Time-Series-Library
git status
git switch main
git pull --ff-only origin main
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

若Git提示本地改动或分支分叉，先保存和处理改动，不要直接强制重置。第二阶段入口已包含在提交 `080890e` 中，后续提交也可使用。数据文件与训练检查点不通过Git同步，继续使用服务器第一阶段的原文件。

## 2. 找到回归主干及其配置

```bash
find outputs/safe_two_stage_cleaned -type f -name best_backbone.pt | sort
```

典型路径如下，其中时间戳以实际输出为准：

```text
outputs/safe_two_stage_cleaned/
└── device58_Informer_L24/
    └── 时间戳/
        ├── config.json
        └── stage1_regression/
            ├── best_backbone.pt
            ├── history.csv
            └── metrics.json
```

设置一个实际检查点路径，再查看对应配置：

```bash
CKPT="$(pwd)/outputs/safe_two_stage_cleaned/device58_Informer_L24/实际时间戳/stage1_regression/best_backbone.pt"
test -f "$CKPT" || { echo "检查点不存在，请修改 CKPT"; exit 1; }
python -m json.tool "$(dirname "$(dirname "$CKPT")")/config.json"
```

确认 `device_id`、`model`、`seq_len`、`d_model`、`n_heads`、`d_ff` 和 `data_root`。默认架构为512/8/2048；若第一阶段使用其他值，第二阶段必须传相同值。

只使用本两阶段入口生成的 `best_backbone.pt`。旧联合分类入口的 `best.pt` 以及第二阶段的 `best_head.pt` 都不能用来代替。

## 3. 单个实验：预检查与启动

下面以设备58、Informer、历史长度24和默认架构为例。其他实验必须相应修改参数。

### 数据预检查，不训练

```bash
python run_safe_two_stage.py \
  --device-id 58 --model Informer --seq-len 24 \
  --stage head --backbone-checkpoint "$CKPT" \
  --data-root "$(pwd)/dataset/fault_selected_cleaned" \
  --output-root "$(pwd)/outputs/safe_stage2_classification"
```

看到 `Preflight only` 表示仅完成数据检查。**该模式检查文件存在、时间连续窗口及样本分布，但不会加载模型，也不会完成检查点参数兼容性验证。** 完整检查点校验会在显式启动后、任何参数更新之前执行。

### 前台启动，仅训练分类头

```bash
python -u run_safe_two_stage.py \
  --device-id 58 --model Informer --seq-len 24 \
  --d-model 512 --n-heads 8 --d-ff 2048 \
  --stage head --backbone-checkpoint "$CKPT" \
  --data-root "$(pwd)/dataset/fault_selected_cleaned" \
  --output-root "$(pwd)/outputs/safe_stage2_classification" \
  --head-learning-rate 0.001 --epochs 100 --patience 3 \
  --train --device cuda:0
```

`--stage head`跳过回归预训练；`--train`开启当前指定阶段的训练。不要省略`--stage head`，默认阶段为`both`。

### 后台启动

```bash
mkdir -p outputs/safe_stage2_classification/logs
LOG="outputs/safe_stage2_classification/logs/device58_Informer_L24_$(date +%Y%m%d_%H%M%S).log"

nohup python -u run_safe_two_stage.py \
  --device-id 58 --model Informer --seq-len 24 \
  --d-model 512 --n-heads 8 --d-ff 2048 \
  --stage head --backbone-checkpoint "$CKPT" \
  --data-root "$(pwd)/dataset/fault_selected_cleaned" \
  --output-root "$(pwd)/outputs/safe_stage2_classification" \
  --head-learning-rate 0.001 --epochs 100 --patience 3 \
  --train --device cuda:0 \
  > "$LOG" 2>&1 &

echo "PID: $!"
echo "日志: $LOG"
tail -f "$LOG"
```

`Ctrl+C`退出日志查看，不停止nohup后台进程。使用`CUDA_VISIBLE_DEVICES=1`时，暴露给进程的第一张卡仍应写`--device cuda:0`。

## 4. 多个回归主干：批量生成分类命令

现有 `run_all.sh` / `run_all_models.sh` 会遍历设备和长度，**不会为每一组自动发现对应主干**，不能给它们传一个公共检查点来启动全部第二阶段。

可以使用下面的文档内辅助代码，按每份回归实验的 `config.json` 生成单独命令。

### 4.1 生成并确认检查点清单

```bash
find outputs/safe_two_stage_cleaned -type f -name best_backbone.pt | sort > head_checkpoints.txt
cat head_checkpoints.txt
```

编辑 `head_checkpoints.txt`，保留本次确实需要使用的检查点，每行一个路径。如果同一设备/模型/长度有多次回归实验，清单会包含多个时间戳；全部保留会分别训练多个分类头。应按预定实验安排或回归验证结果选择，不按测试指标挑选。

### 4.2 生成脚本，不自动执行

在仓库根目录执行以下代码。它读取清单和JSON配置，不加载检查点权重、不训练：

```bash
python - <<'PY'
import json
import shlex
import sys
from pathlib import Path

root = Path.cwd()
assert (root / 'run_safe_two_stage.py').is_file(), '请先进入仓库根目录'
# 如果数据存放在其他目录，只修改此处。
data_root = root / 'dataset/fault_selected_cleaned'
output_root = root / 'outputs/safe_stage2_classification'
paths = [Path(line.strip()).resolve()
         for line in Path('head_checkpoints.txt').read_text().splitlines()
         if line.strip()]
assert paths, '检查点清单为空'
assert len(paths) == len(set(paths)), '检查点清单存在重复路径'
commands = ['#!/usr/bin/env bash', 'set -euo pipefail',
            'cd -- ' + shlex.quote(str(root))]
for checkpoint in paths:
    assert checkpoint.is_file(), checkpoint
    cfg = json.loads((checkpoint.parent.parent / 'config.json').read_text())
    assert (data_root / f"设备{cfg['device_id']}_cleaned.csv").is_file()
    command = [sys.executable, '-u', str(root / 'run_safe_two_stage.py'),
               '--device-id', str(cfg['device_id']), '--model', cfg['model'],
               '--seq-len', str(cfg['seq_len']),
               '--d-model', str(cfg['d_model']), '--n-heads', str(cfg['n_heads']),
               '--d-ff', str(cfg['d_ff']), '--seed', str(cfg['seed']),
               '--regression-loss', cfg['regression_loss'],
               '--head-hidden', str(cfg.get('head_hidden', 16)),
               '--head-dropout', str(cfg.get('head_dropout', 0.1)),
               '--batch-size', str(cfg['batch_size']),
               '--data-root', str(data_root), '--output-root', str(output_root),
               '--stage', 'head', '--backbone-checkpoint', str(checkpoint),
               '--head-learning-rate', '0.001', '--epochs', '100',
               '--patience', '3', '--device', 'cuda:0']
    # 不固定--train：运行脚本时显式传入；默认仍为预检查。
    commands.append(shlex.join(command) + ' "$@"')
Path('run_selected_heads.sh').write_text('\n'.join(commands) + '\n')
print(f'已生成 {len(paths)} 组命令：run_selected_heads.sh；尚未启动训练。')
PY
```

### 4.3 先预检查，再后台执行

```bash
bash -n run_selected_heads.sh
bash run_selected_heads.sh

mkdir -p outputs/safe_stage2_classification/logs
LOG="outputs/safe_stage2_classification/logs/all_heads_$(date +%Y%m%d_%H%M%S).log"
nohup bash run_selected_heads.sh --train > "$LOG" 2>&1 &
echo "PID: $!"
tail -f "$LOG"
```

按清单顺序执行，每组使用自己的主干；失败即停止。脚本没有自动断点续跑功能，重复执行会创建新的分类实验目录。若某组失败，解决问题后从清单中移除已完成项，再重新生成脚本。

## 5. 训练参数与冻结检查

- 当前分类头：`Linear(4,16) → GELU → Dropout(0.1) → Linear(16,1)`，97个参数。尚未引入此前讨论的隐藏表示辅助输入。
- 主干参数关闭梯度、前向使用 `no_grad()`、保持 `eval()`；分类头单独处于训练模式。
- 分类损失：`BCEWithLogitsLoss(pos_weight=训练负例数/训练正例数)`；阈值默认0.5。
- `--epochs`是分类阶段最多epoch数；`--head-learning-rate`控制头的初始学习率，随后按现有type1策略衰减；`--patience`按验证BCE早停。
- `--learning-rate`是回归阶段参数，在head-only模式下不会决定分类头学习率。
- 冻结后的主干参数和buffer在每个epoch及最佳分类检查点重载后均校验一致性。Informer原ProbAttention仍可能随机采样，这不表示参数变化。

## 6. 输出、汇总与完成判据

本文命令使用独立输出根目录，不覆盖原回归实验：

```text
outputs/safe_stage2_classification/
└── device58_Informer_L24/新时间戳/
    ├── config.json
    ├── summary.json
    └── stage2_head/
        ├── best_head.pt
        ├── history.csv
        ├── metrics.json
        ├── predictions.csv
        └── freeze_audit.json
```

`config.json`记录主干来源、最佳回归epoch、主干SHA256、类别权重及数据统计。`best_head.pt`包含完整冻结主干和最佳分类头参数。

训练正常完成应有 `metrics.json`、`predictions.csv` 和 `summary.json`；检查 `freeze_audit.json` 中 `parameters_and_buffers_unchanged` 为 `true`。仅看到某个checkpoint文件不代表整组评估已经完成。

汇总分类实验：

```bash
python scripts/long_term_forecast/Fault_script/safe_two_stage_cleaned/collect_results.py \
  --output-root outputs/safe_stage2_classification
```

生成 `comparison.csv`。head-only实验没有重新执行回归测试，对应MSE/MAE列为空；原回归指标保留在源实验的 `stage1_regression/metrics.json`。

## 7. 常见问题

| 提示或现象 | 处理方式 |
| --- | --- |
| `Incompatible backbone checkpoint fields` | 按错误中的字段核对设备、模型、长度、架构。`input_sha256`不一致说明CSV字节内容变化，应恢复第一阶段数据或选对应检查点，不绕过校验。 |
| `Expected a stage1 regression checkpoint` | 选择本两阶段入口生成的 `best_backbone.pt`，不要使用旧联合分类或第二阶段检查点。 |
| `CUDA unavailable` | 激活第一阶段的CUDA PyTorch环境，并检查GPU可见性。 |
| 显存不足 | 降低 `--batch-size`，例如8；batch size不属于主干兼容性限制。 |
| 只显示 `Preflight only` | 未传`--train`，这时不会训练。 |
| 重新出现回归阶段日志 | 检查命令是否明确传入`--stage head`，是否误用了旧批量脚本。 |
| 只有某些模型能加载 | 确认每份checkpoint对应自己的`--model`，不能把设备默认路由当作所有对比实验的模型。 |

第二阶段可直接使用现有回归主干，不需要重新跑第一阶段。建议先完成一个实验确认环境和输出，再启动整个清单。
