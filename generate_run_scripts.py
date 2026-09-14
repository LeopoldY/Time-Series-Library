import os
import stat

# ================= 配置区域 =================
# 脚本保存路径
SCRIPT_DIR = "scripts/long_term_forecast/Fault_script/yc_tools/run_classification/"
# 数据集所在路径
DATA_ROOT = "./dataset/fault_selected/"

# 公共参数 (基于您之前的修改：任务名为 long_term_forecast 但实际执行分类逻辑)
COMMON_ARGS = (
    "--task_name long_term_forecast "
    "--is_training 1 "
    f"--root_path {DATA_ROOT} "
    "--data custom "
    "--features MS "
    "--seq_len 96 "    # 历史窗口
    "--label_len 48 "  # Informer 需要 label_len
    "--pred_len 1 "    # 预测未来 1 个点 (分类结果)
    "--c_out 2 "       # 二分类
    "--enc_in 7 "      # 【注意】请根据您CSV实际特征列数修改此值
    "--dec_in 7 "      # 同上
    "--des 'Exp' "
    "--itr 1 "
    "--batch_size 16 "
    "--learning_rate 0.001 "
)

# 模型特定参数配置
MODEL_CONFIGS = {
    # 1. TimesNet: 适合捕捉周期性，参数重点是 top_k
    "TimesNet": (
        "--model TimesNet "
        "--e_layers 2 --d_layers 1 "
        "--d_model 32 --d_ff 32 "
        "--top_k 3 "
    ),
    
    # 2. Informer: 适合长序列，但在短预测/分类中需防止稀疏采样崩溃
    "Informer": (
        "--model Informer "
        "--e_layers 2 --d_layers 1 "
        "--d_model 512 --n_heads 8 "
        "--factor 1 "          # 关键：设为1退化为全注意力，防止短序列报错
        "--output_attention "  # 强制输出注意力，通常能规避部分bug
    ),
    
    # 3. Nonstationary_Transformer: 适合非平稳数据 (故障数据通常非平稳)
    "Nonstationary_Transformer": (
        "--model Nonstationary_Transformer "
        "--e_layers 2 --d_layers 1 "
        "--d_model 512 --n_heads 8 "
        "--p_hidden_dims 128 128 " # 投影层维度
        "--d_ff 2048 "
    ),
    
    # 4. PatchTST: SOTA模型，基于Patch (补充的第四个模型)
    "PatchTST": (
        "--model PatchTST "
        "--e_layers 3 --n_heads 4 "
        "--d_model 128 --d_ff 256 "
        "--patch_len 16 --stride 8 " # Patch设置
        "--dropout 0.1 "
    )
}
# ===========================================

def main():
    # 1. 创建目录
    if not os.path.exists(SCRIPT_DIR):
        os.makedirs(SCRIPT_DIR)
        print(f"Created directory: {SCRIPT_DIR}")

    # 2. 遍历生成 4 个 .sh 文件
    for model_name, specific_args in MODEL_CONFIGS.items():
        file_name = f"{model_name}.sh"
        file_path = os.path.join(SCRIPT_DIR, file_name)
        
        # 构建 Shell 脚本内容
        # 使用 Loop 遍历数据目录下所有 csv 文件
        script_content = f"""#!/bin/bash

# Auto-generated script for {model_name}
# Task: Fault Classification (via Modified Long Term Forecast)

# 显式设置 Python 路径，防止环境问题
# export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "Start Training Model: {model_name}"
echo "=========================================="

# 遍历 {DATA_ROOT} 下的所有 .csv 文件
for file in {DATA_ROOT}*.csv
do
    # 获取文件名 (例如 device1.csv)
    data_path=$(basename "$file")
    # 生成唯一的 model_id (去掉 .csv 后缀)
    device_id="${{data_path%.*}}"
    
    echo ""
    echo ">>> Processing Device: $device_id"
    
    python -u run.py \\
      --data_path "$data_path" \\
      --model_id "Fault_{model_name}_$device_id" \\
      {specific_args} \\
      {COMMON_ARGS}
      
    if [ $? -ne 0 ]; then
        echo "Error running {model_name} on $device_id"
    fi
done

echo "Done."
"""
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 3. 赋予执行权限 (chmod +x)
        st = os.stat(file_path)
        os.chmod(file_path, st.st_mode | stat.S_IEXEC)
        
        print(f"Generated script: {file_path} (Executable)")

if __name__ == "__main__":
    main()