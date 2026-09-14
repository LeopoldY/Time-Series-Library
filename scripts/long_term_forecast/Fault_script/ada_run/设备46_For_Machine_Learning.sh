#!/bin/bash

# Auto-generated for 设备46_For_Machine_Learning (Rich Features, enc_in=7)
export CUDA_VISIBLE_DEVICES=1

root_path='/mnt/sdc1/skx/Time-Series-Library/dataset/fault_raw/processed_rich'
data_path='设备46_For_Machine_Learning.csv'
device_id='设备46_For_Machine_Learning'
model_name='PatchTST'
seq_lens=(3 6 12 24)

for seq_len in "${seq_lens[@]}"; 
do
  echo ">>> EXP_INFO: Device:$device_id Model:$model_name Seq:$seq_len"
  if [ "$seq_len" -eq "3" ]; then
    patch_len=1; stride=1
  else
    patch_len=$(expr $seq_len / 2)
    stride=$(expr $patch_len / 2)
  fi
  python -u run.py \
    --model_id $device_id'_'$seq_len'_'rich_SAFE \
    --model PatchTST \
    --seq_len $seq_len \
    --patch_len $patch_len \
    --stride $stride \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --batch_size 16 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --data custom \
    --features MS \
    --target w_level4 \
    --label_len 1 \
    --pred_len 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 1 \
    --des 'Exp_SAFE_Rich' \
    --itr 1
done
