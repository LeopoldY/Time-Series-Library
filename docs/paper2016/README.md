# 钟将等（2016）清洗与逐设备十折训练

来源：《基于告警日志的网络故障预测》，计算机应用 36(S1):49–53，§1.2（p50）、§3.2.2（p52）。论文内容仅作方法依据。

## 清洗

输入 `dataset/fault_raw/raw_data` 全部87台设备，已核对与工作区 `fault-forecast-dataset/csv` 的87份CSV逐表一致，不将重复副本拼接。保留原文件，输出到独立 `dataset/fault_paper2016`。

1. 沿用空白转空值、删除全空列。
2. 删除 `告警类型=衍生告警`。当前输入只有普通告警和根源告警，本次删除0条；不依据名称猜测因果关系。
3. 删除 `0 ≤ Repair_Interval < 22` 秒的闪断告警，22秒保留。负数/缺失/非有限时长按未知保留并审计，本批保留的未知时长为0。
4. 按发生时间稳定排序，以（设备、级别、名称、告警类型）为重复键，与最近一条**已保留**告警比较；间隔小于提示900、次要240、重要180、紧急120秒时删除，恰等于阈值保留。论文没有写明重复键、链式规则及阈值边界，这些是可复现的实施约定。
5. 沿用旧四设备清洗流程的统一小时起点（全设备原始最早时间）、左闭右开窗口、中点时间戳和补零。四级同时全零连续≥336小时，整个区间删除；保存原时间戳，训练窗口不得跨越断点。

结果：149,761条 → 104,469条；闪断删除25,605条，时间去重删除19,687条；删除366,011个全零小时，保留521,737小时行。`cleaning_audit.csv` 包含逐设备数量和输入/输出SHA256；`summary.json` 为总计。

`events/` 是事件清洗结果，`four_levels/` 是训练小时表，`folds/` 是逐设备每行的基础折编号。数据沿用仓库忽略规则，不上传GitHub；服务器从原始CSV重建。

## 十折划分及适配界限

论文写明“十轮交叉验证”，没有给出随机种子、是否打乱、验证集、去重键；设备实验没有另给划分细节。本实现将该交叉验证方法应用到每台设备，采用确定性的**按时间排序分十块**：第k块测试，第(k+1)%10块验证，其余八块训练，k=0..9。比例约80/10/10；验证块是神经网络早停所需的工程扩展，不宣称完全复现论文未公开的划分。

历史输入、目标小时及解码器时间特征必须全部属于同一集合，且小时连续。这样训练、验证、测试没有共用观测行；边界的前L个目标被剔除。因此十轮覆盖所有基础测试块，但不覆盖每个边界目标。该方法是离线交叉验证，训练可包含测试时段之后的数据，不等同于只用过去预测未来的滚动回测。

每折只用训练行提取设备12维画像并拟合StandardScaler/KMeans；路由随折重新计算。保持已有SAFE四级下一小时计数回归＋“任一级别有告警”二分类联合训练，历史长度3/6/12/24。论文以紧急故障为目标并使用两级时间窗，**本次只借鉴其清洗和划分方法，没有将当前模型或标签替换为论文模型**。基于清除时长的离线清洗依赖事后信息，指标代表事后清洗日志，不能直接视为在线性能。

预检87×4×10=3,480组：3,458组可训练；22组因空集合或训练单一类别跳过，并写入每折 `skipped.jsonl`（不会静默终止后续设备）。详细见 `split_audit.csv`。不能用单一类别训练集估计当前正负类加权BCE；不人为补造样本。

## 服务器复现与启动

在仓库根目录、已安装适配服务器CUDA的PyTorch环境中：

```bash
git fetch origin
git switch codex/paper2016-cleaning-cv
git pull --ff-only origin codex/paper2016-cleaning-cv
python -m pip install -r scripts/long_term_forecast/Fault_script/safe_binary_cleaned/requirements.txt
python scripts/tools/prepare_paper2016.py
python scripts/tools/preflight_paper2016.py
mkdir -p outputs/safe_paper2016
nohup bash scripts/tools/train_paper2016.sh --device cuda:0 \
  > outputs/safe_paper2016/train.log 2>&1 &
echo $!
```

原始文件必须在 `dataset/fault_raw/raw_data/`。若路径不同，清洗入口支持 `--input-dir`；训练脚本支持 `DATA_ROOT`、`OUTPUT_ROOT`、`PYTHON_BIN` 环境变量。十折串行执行，每次训练独立初始化、验证早停、重载最佳权重后测试。输出 `outputs/safe_paper2016/foldN/时间戳/`，保留模型、配置、各折路由、预测及指标；不要按测试指标挑折或挑模型。

快速服务器冒烟测试（3台设备、1折、1个长度、1轮）：

```bash
python run_safe_joint_routed_training.py --data-root dataset/fault_paper2016/four_levels \
  --output-root outputs/safe_paper2016_smoke --cv-fold 0 --devices 27 58 83 \
  --lengths 3 --epochs 1 --d-model 16 --n-heads 2 --d-ff 32 --device cuda:0 --train
```

## 本地验证

- 全87台源数据一致性、清洗计数守恒、所有输出CSV回读验证。
- 全3,480组预检；阈值边界、去重锚点、幂等性、十折观测行不交叠、时间断点过滤的单元测试。
- `python -m unittest discover -s tests -p 'test_paper2016.py' -v`
- 18项既有SAFE测试全部通过；3种主干各完成1轮CPU真实训练、验证早停保存/重载、测试预测导出（设备27/58/83、fold0、L3、小架构）。详见 `validation.json`。尚未进行完整CUDA训练，冒烟结果不代表收敛性能。
