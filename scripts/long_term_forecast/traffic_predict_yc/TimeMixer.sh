export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# 针对你的数据集配置
root_path_conf="./dataset/traffic_predict/"
data_path_conf="cleaned_traffic_data.csv"
enc_in_conf=90
dec_in_conf=90
c_out_conf=90

target_name="L0110-L0101"

# TimeMixer 特定参数
seq_len=24
pred_len=12
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8 # 显存如果不够可以调小

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id Traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --target $target_name \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window