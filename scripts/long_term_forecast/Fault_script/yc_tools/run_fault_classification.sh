#!/bin/bash

# 配置路径
root_path_name=./dataset/fault_selected/
# 请确保该目录下有您的 csv 文件，例如 data1.csv, data2.csv 等

# 设置序列长度和预测长度
seq_len=24
pred_len=1

# 设置特征数量 (Encoder Input Size)，请根据您的CSV实际特征列数修改
enc_in=7 

# 模型列表
models_name=("Nonstationary_Transformer" "TimesNet" "Informer")

# 遍历目录下的所有csv文件
for file in ./dataset/fault_selected/*.csv
do
    data_path_name=$(basename "$file")
    echo "Processing data: $data_path_name"

    for model_name in "${models_name[@]}"
    do 
        echo "Training model: $model_name"

        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id fault_detect_${model_name}_${seq_len}_${pred_len} \
          --model $model_name \
          --data custom \
          --features MS \
          --seq_len $seq_len \
          --label_len 48 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in $enc_in \
          --dec_in $enc_in \
          --c_out 2 \
          --des 'Exp' \
          --itr 1
    done
done