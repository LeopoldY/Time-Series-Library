export CUDA_VISIBLE_DEVICES=1

model_name=PatchTST

# ==========================================
# 1. 数据集适配配置
# ==========================================
root_path_conf="./dataset/traffic_predict/"
data_path_conf="cleaned_traffic_data.csv"
target_conf="L0110-L0101" # 必须指定一个存在的列名

# 维度配置 (90列数据)
enc_in_conf=90
dec_in_conf=90
c_out_conf=90

# ==========================================
# 2. 任务执行
#    注意：由于总数据只有约300行，seq_len设为24。
#    因此无法运行官方的 pred_len 96/192/336 等任务。
#    这里改为适配您数据的短预测任务：pred_len 6 和 12。
# ==========================================

# 任务 1: seq_len 24 -> pred_len 6 (超短时预测)
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_24_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.001 \
  --patch_len 8 \
  --stride 4 \
  --target $target_conf \
  --itr 1

# 任务 2: seq_len 24 -> pred_len 12 (短时预测)
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_conf \
  --data_path $data_path_conf \
  --model_id traffic_24_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_conf \
  --dec_in $dec_in_conf \
  --c_out $c_out_conf \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 0.001 \
  --patch_len 8 \
  --stride 4 \
  --target $target_conf \
  --itr 1

# 以下是官方标准长度任务，因数据量不足 (300行 < 192+96) 暂时注释掉
# 如果未来数据量增加到 1000 行以上，可以取消注释运行
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_path_conf \
#   --data_path $data_path_conf \
#   --model_id traffic_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   ... (其余参数同上)