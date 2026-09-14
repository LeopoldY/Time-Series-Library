#!/bin/bash

# 设置可见的CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 定义序列长度数组，元素之间用空格分隔
seq_lens=(3 6 12 24)

root_path='/mnt/sdc1/skx/Time-Series-Library/dataset/fault/S9300_S9306/设备27'
device_id=S9300_S9306_设备27
freq=60
freq_str=h
data_path=window_$freq'_'data2.csv


#iTransformer
for seq_len in "${seq_lens[@]}"; 
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $device_id'_'$seq_len'_'1_scale_false_ \
      --model iTransformer \
      --data custom \
      --features MS \
      --target w_level4 \
      --seq_len $seq_len \
      --label_len 1 \
      --pred_len 1 \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 4 \
      --dec_in 4 \
      --c_out 1 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --freq $freq_str \
      --itr 1
done
