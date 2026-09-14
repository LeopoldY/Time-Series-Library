#!/bin/bash

# Auto-generated for 设备3 (Type: Intermittent -> Model: iTransformer)

root_path='/mnt/sdc1/skx/Time-Series-Library/dataset/fault/S5300_S5328C-EI/设备3'
data_path='window_60_data2.csv'
device_id='设备3'
model_name='iTransformer'
seq_lens=(3 6 12 24)

for seq_len in "${seq_lens[@]}"; 
do
  echo ">>> EXP_INFO: Device:$device_id Model:$model_name Seq:$seq_len"
  python -u run.py \
    --model_id $device_id'_'$seq_len'_'SAFE \
    --model iTransformer \
    --seq_len $seq_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --batch_size 32 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --data custom \
    --features MS \
    --target w_level4 \
    --label_len 1 \
    --pred_len 1 \
    --enc_in 4 \
    --dec_in 4 \
    --c_out 1 \
    --des 'Exp_SAFE' \
    --itr 1
done
