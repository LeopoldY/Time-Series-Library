export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

# 针对你的数据集配置
root_path_conf="./dataset/traffic_predict/"
data_path_conf="cleaned_traffic_data.csv"
enc_in_conf=90  # 你的数据有90列
dec_in_conf=90
c_out_conf=90

target_name="L0110-L0101"

seq_len=8
label_len=4
pred_len=4


# 任务1: 预测未来 96 个时间步
# 注意: 如果你的数据量太少(只有300行)，建议将 seq_len 改为 24, pred_len 改为 12 进行测试
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_prediction \
  --model $model_name \
  --data custom \
  --features M \
  --target $target_name \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --itr 1