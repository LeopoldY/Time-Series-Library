import os
import stat
import glob
import pandas as pd

def find_csv_files(dataset_root):
    """递归查找所有 CSV"""
    csv_files = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                path_parts = root.strip(os.sep).split(os.sep)
                if len(path_parts) >= 2:
                    device_name = path_parts[-1]
                    category = path_parts[-2]
                else:
                    device_name = "unknown"
                    category = "unknown"
                csv_files.append((full_path, device_name, category))
    print(f"[Data Discovery] Found {len(csv_files)} CSV files in {dataset_root}")
    return csv_files

def generate_individual_scripts(df_clusters, output_dir, strategy_mapping):
    """
    生成 .sh 脚本 (带日志埋点，方便后续自动提取 CSV)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for _, row in df_clusters.iterrows():
        device_name = row['Device_Name']
        device_label = row['Label']
        root_path = row['Root_Path']
        data_path = row['Data_Path']
        
        # 自适应选择模型
        model_name = 'Transformer'
        for key, val in strategy_mapping.items():
            if key in device_label:
                model_name = val
                break
        
        script_path = os.path.join(output_dir, f"{device_name}.sh")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# Auto-generated for {device_name} (Type: {device_label} -> Model: {model_name})\n\n")
            
            f.write(f"root_path='{root_path}'\n")
            f.write(f"data_path='{data_path}'\n")
            f.write(f"device_id='{device_name}'\n")
            f.write(f"model_name='{model_name}'\n")
            f.write("seq_lens=(3 6 12 24)\n\n") 
            
            # 通用参数
            common_params = (
                "    --task_name long_term_forecast \\\n"
                "    --is_training 1 \\\n"
                "    --root_path $root_path \\\n"
                "    --data_path $data_path \\\n"
                "    --data custom \\\n"
                "    --features MS \\\n"
                "    --target w_level4 \\\n"
                "    --label_len 1 \\\n"
                "    --pred_len 1 \\\n"
                "    --enc_in 4 \\\n"
                "    --dec_in 4 \\\n"
                "    --c_out 1 \\\n"
                "    --des 'Exp_SAFE' \\\n"
                "    --itr 1"
            )

            f.write("for seq_len in \"${seq_lens[@]}\"; \n")
            f.write("do\n")
            
            # === 关键修改：打印便于解析的日志标记 ===
            f.write("  echo \">>> EXP_INFO: Device:$device_id Model:$model_name Seq:$seq_len\"\n")
            # =======================================

            if model_name == 'PatchTST':
                f.write("  if [ \"$seq_len\" -eq \"3\" ]; then\n")
                f.write("    patch_len=1\n")
                f.write("    stride=1\n")
                f.write("  else\n")
                f.write("    patch_len=$(expr $seq_len / 2)\n")
                f.write("    stride=$(expr $patch_len / 2)\n")
                f.write("  fi\n\n")
                
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --patch_len $patch_len \\\n")
                f.write(f"    --stride $stride \\\n")
                f.write(f"    --e_layers 2 \\\n")
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 16 \\\n") 
                f.write(common_params + "\n")

            elif model_name == 'Informer':
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --e_layers 3 \\\n") 
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 32 \\\n")
                f.write(common_params + "\n")
                
            else: # iTransformer / TimesNet
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --e_layers 2 \\\n")
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 32 \\\n")
                f.write(common_params + "\n")

            f.write("done\n") 

        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)
        count += 1

    print(f"[Script Gen] Generated {count} scripts in {output_dir}")

# =========================================================
# [NEW FUNCTION] Added for Raw Data Experiment (Rich Features)
# =========================================================
def generate_scripts_dynamic(df_clusters, output_dir, strategy_mapping, enc_in=4):
    """
    生成支持动态特征维度(enc_in)的实验脚本。
    用于处理包含告警类型、修复时长等丰富特征的数据集。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for _, row in df_clusters.iterrows():
        device_name = row['Device_Name']
        device_label = row['Label']
        root_path = row['Root_Path']
        data_path = row['Data_Path']
        
        # 1. 确定模型
        model_name = 'Transformer'
        for key, val in strategy_mapping.items():
            if key in device_label:
                model_name = val
                break
        
        script_path = os.path.join(output_dir, f"{device_name}.sh")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# Auto-generated for {device_name} (Rich Features, enc_in={enc_in})\n")

            f.write("export CUDA_VISIBLE_DEVICES=1\n\n")
            
            f.write(f"root_path='{root_path}'\n")
            f.write(f"data_path='{data_path}'\n")
            f.write(f"device_id='{device_name}'\n")
            f.write(f"model_name='{model_name}'\n")
            f.write("seq_lens=(3 6 12 24)\n\n") 
            
            # 动态参数配置
            # 注意: c_out=1 (只预测 w_level4), enc_in/dec_in=动态传入
            common_params = (
                "    --task_name long_term_forecast \\\n"
                "    --is_training 1 \\\n"
                "    --root_path $root_path \\\n"
                "    --data_path $data_path \\\n"
                "    --data custom \\\n"
                "    --features MS \\\n"
                "    --target w_level4 \\\n"
                "    --label_len 1 \\\n"
                "    --pred_len 1 \\\n"
                f"    --enc_in {enc_in} \\\n"  # <--- 动态参数
                f"    --dec_in {enc_in} \\\n"  # <--- 动态参数
                "    --c_out 1 \\\n"
                "    --des 'Exp_SAFE_Rich' \\\n"
                "    --itr 1"
            )

            f.write("for seq_len in \"${seq_lens[@]}\"; \n")
            f.write("do\n")
            f.write("  echo \">>> EXP_INFO: Device:$device_id Model:$model_name Seq:$seq_len\"\n")
            
            if model_name == 'PatchTST':
                f.write("  if [ \"$seq_len\" -eq \"3\" ]; then\n")
                f.write("    patch_len=1; stride=1\n")
                f.write("  else\n")
                f.write("    patch_len=$(expr $seq_len / 2)\n")
                f.write("    stride=$(expr $patch_len / 2)\n")
                f.write("  fi\n")
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'rich_SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --patch_len $patch_len \\\n")
                f.write(f"    --stride $stride \\\n")
                f.write(f"    --e_layers 2 \\\n")
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 16 \\\n") 
                f.write(common_params + "\n")

            elif model_name == 'Informer':
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'rich_SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --e_layers 3 \\\n") 
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 32 \\\n")
                f.write(common_params + "\n")
                
            else: 
                f.write("  python -u run.py \\\n")
                f.write(f"    --model_id $device_id'_'$seq_len'_'rich_SAFE \\\n")
                f.write(f"    --model {model_name} \\\n")
                f.write(f"    --seq_len $seq_len \\\n")
                f.write(f"    --e_layers 2 \\\n")
                f.write(f"    --d_layers 1 \\\n")
                f.write(f"    --factor 3 \\\n")
                f.write(f"    --batch_size 32 \\\n")
                f.write(common_params + "\n")

            f.write("done\n") 

        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)
        count += 1

    print(f"[Script Gen] Generated {count} scripts (enc_in={enc_in}) in {output_dir}")