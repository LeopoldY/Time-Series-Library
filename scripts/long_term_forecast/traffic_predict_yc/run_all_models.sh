#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=1

# ==========================================
# 1. 数据配置
# ==========================================
root_path_conf="./dataset/traffic_predict/"
data_path_conf="cleaned_traffic_data.csv"
target_conf="L0110-L0101"  # 指定第一列作为Target锚点

# 维度配置 (根据您的90列数据)
enc_in_conf=90
dec_in_conf=90
c_out_conf=90

# 序列长度配置 (关键修改：调小以适应小数据集)
seq_len_conf=24    # 输入过去 24 个点
pred_len_conf=12   # 预测未来 12 个点
label_len_conf=12  # label长度 (seq_len的一半)

# 训练通用参数
train_epochs=3     # 数据少，跑几轮即可
batch_size=8       # 小Batch

echo "========================================================"
echo "Start Running All Models for Small Traffic Dataset..."
echo "Seq Len: $seq_len_conf, Pred Len: $pred_len_conf"
echo "========================================================"

# ==========================================
# 2. 运行 iTransformer
# ==========================================
model_name=iTransformer
echo ">>> Running $model_name ..."

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_iTransformer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len_conf \
  --label_len $label_len_conf \
  --pred_len $pred_len_conf \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size $batch_size \
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --target $target_conf \
  --itr 1

echo ">>> $model_name Finished!"
echo "--------------------------------------------------------"

# ==========================================
# 3. 运行 TimeMixer
# ==========================================
model_name=TimeMixer
echo ">>> Running $model_name ..."

# TimeMixer 特有参数
down_sampling_layers=3
down_sampling_window=2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_TimeMixer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len_conf \
  --label_len 0 \
  --pred_len $pred_len_conf \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --batch_size $batch_size \
  --learning_rate 0.01 \
  --train_epochs $train_epochs \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --target $target_conf \
  --itr 1

echo ">>> $model_name Finished!"
echo "--------------------------------------------------------"

# ==========================================
# 4. 运行 TimeXer
# ==========================================
model_name=TimeXer
echo ">>> Running $model_name ..."

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_TimeXer \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len_conf \
  --label_len $label_len_conf \
  --pred_len $pred_len_conf \
  --e_layers 2 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size $batch_size \
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --target $target_conf \
  --itr 1

echo ">>> $model_name Finished!"
echo "========================================================"
echo "All Models Processed Successfully."