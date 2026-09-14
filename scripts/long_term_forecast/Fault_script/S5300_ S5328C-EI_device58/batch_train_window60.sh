#!/bin/bash

# 设置可见的CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 定义序列长度数组，元素之间用空格分隔
seq_lens=(3 6 12 24)

root_path='/mnt/sdc1/skx/Time-Series-Library/dataset/fault/S5300_S5328C-EI/设备58'
device_id=S5300_S5328C-EI_设备58
freq=60
freq_str=h
data_path=window_$freq'_'data2.csv

# #Informer
for seq_len in "${seq_lens[@]}"; 
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $device_id'_'$seq_len'_'1 \
      --model Informer \
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
      --freq $freq_str \
      --itr 1
done

#Dlinear
for seq_len in "${seq_lens[@]}"; 
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $device_id'_'$seq_len'_'1 \
      --model DLinear \
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
      --freq $freq_str \
      --itr 1
done

#iTransformer
for seq_len in "${seq_lens[@]}"; 
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $device_id'_'$seq_len'_'1 \
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

#PatchTST
for seq_len in "${seq_lens[@]}"; 
do
  patch_len=6
  stride=3
  if [ "$seq_len" -eq "3" ]; then
    patch_len=1
    stride=1
  else
    patch_len=$(expr $seq_len / 2)
    stride=$(expr $patch_len / 2)
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $device_id'_'$seq_len'_'$patch_len'_'$stride'_'1 \
    --model PatchTST \
    --data custom \
    --features MS \
    --target w_level4 \
    --seq_len $seq_len \
    --label_len 1 \
    --pred_len 1 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 4 \
    --dec_in 4 \
    --c_out 1 \
    --des 'Exp' \
    --patch_len $patch_len \
    --stride $stride \
    --batch_size 16 \
    --freq $freq_str \
    --itr 1
done


# TimesNet
for seq_len in "${seq_lens[@]}"; 
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $device_id'_'$seq_len'_top_k_'$(expr $seq_len / 2)'_1' \
    --model TimesNet \
    --data custom \
    --features MS \
    --target w_level4 \
    --seq_len $seq_len \
    --label_len 1 \
    --pred_len 1 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 4 \
    --dec_in 4 \
    --c_out 1 \
    --d_model 256 \
    --d_ff 512 \
    --top_k $(expr $seq_len / 2) \
    --des 'Exp' \
    --freq $freq_str \
    --itr 1
done
