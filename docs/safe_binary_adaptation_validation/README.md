# SAFE 二分类适配验证记录

日期：2026-09-14。本次未启动训练入口、未调用optimizer.step、未产生训练检查点或训练结果目录。

## 实现范围

新增独立 `run_safe_binary.py`、`data_provider/safe_binary_data.py`、`models/SAFE_Binary.py`，以及 `scripts/long_term_forecast/Fault_script/safe_binary_cleaned/` 内批量脚本、单设备脚本、依赖清单和使用说明。所有原有受版本控制的文件保持不变。

分类头为4→16→1，GELU及Dropout，97个参数。标签为下一小时任一级别告警出现。路由固定为27/69→iTransformer、58→Informer、83→PatchTST。

## 已执行的验证

- Python语法及全部6份Shell脚本语法检查通过。
- `unittest` 三项测试通过，其中模型测试覆盖三种骨干×四种历史长度，共12种配置；使用缩小的模型宽度进行人工输入前向、BCE梯度以及检查点往返验证，无优化器更新。
- 人工数据验证四个级别各自都能产生正例、全零为负例、标签取下一行且相隔一小时、未来计数未输入解码器、时间切分不混用标签、窗口不会跨断点。
- 三种骨干×四种历史长度，以默认d_model=512/n_heads=8/d_ff=2048进行batch=1前向检查，全部通过。
- 无正例指标边界检查通过，Recall/F1/AP/ROC-AUC按约定返回null。
- 直接执行 `run_all.sh` 的默认预检查模式，16组设备/历史长度组合全部成功，没有进入训练分支。
- 实际数据48组“设备×历史长度×数据划分”的窗口审计全部完成，各组均含正负例。完整统计见 `window_audit.csv`。

## 验证边界

本地CPU验证环境：Python 3.12、PyTorch 2.14.0，其他分析依赖来自临时环境。未验证服务器CUDA执行、完整训练收敛或最终性能；使用说明声明torch>=2.1，具体服务器环境应先执行默认预检查及单元测试。未将本地临时依赖目录写入提交代码或脚本。

原Informer的ProbAttention在eval模式下仍有随机抽样，检查点一致性测试使用相同随机种子；此行为保留原模型实现。
