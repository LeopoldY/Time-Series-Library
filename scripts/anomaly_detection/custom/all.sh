export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model Autoformer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model Crossformer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model DLinear \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 100 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model ETSformer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 100 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --d_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model FEDformer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model Informer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name multi_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/anomaly/ \
  --model_id MULTI_TEST \
  --model iTransformer \
  --data MULTI_ANOMALY \
  --features M \
  --seq_len 200 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 2 \
  --c_out 2 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10