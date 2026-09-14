import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ================= 配置区域 =================
# UEA 数据集根目录
UEA_ROOT = "dataset/UEA/"

# 要运行的模型列表
MODELS = ['TimesNet', 'Informer', 'PatchTST', 'iTransformer']

# 序列长度 (必须与转换时设置的一致)
SEQ_LEN = 96 

# 训练参数
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# ===========================================

def get_enc_in(device_path, device_name):
    """自动从 .ts 文件读取特征维度 (enc_in)"""
    ts_file = os.path.join(device_path, f"{device_name}_TRAIN.ts")
    try:
        with open(ts_file, 'r') as f:
            for line in f:
                if line.startswith("@dimensions"):
                    return int(line.split()[1])
    except:
        return 0
    return 0

def main():
    # 扫描 dataset/UEA/ 下的所有子文件夹（即设备名）
    device_paths = glob.glob(os.path.join(UEA_ROOT, "*"))
    # 过滤掉非文件夹
    device_paths = [d for d in device_paths if os.path.isdir(d)]
    
    print(f"Found {len(device_paths)} datasets (devices).")
    
    for device_path in device_paths:
        device_name = os.path.basename(device_path)
        
        # 获取该设备的特征维度
        enc_in = get_enc_in(device_path, device_name)
        if enc_in == 0:
            print(f"[Skip] Could not determine dimensions for {device_name}")
            continue
            
        print(f"\n=======================================================")
        print(f"Processing Device: {device_name} (Dims={enc_in})")
        print(f"=======================================================")
        
        for model in MODELS:
            print(f"\n>>> Training Model: {model} ...")
            
            cmd = (
                f"python -u run.py \\\n"
                f"  --task_name classification \\\n"
                f"  --is_training 1 \\\n"
                f"  --root_path {UEA_ROOT}/{device_name} \\\n"  # 根目录
                f"  --data_path {device_name} \\\n" # 设备名(文件夹名)
                f"  --model_id {device_name}_{model}_{SEQ_LEN} \\\n"
                f"  --model {model} \\\n"
                f"  --data UEA \\\n"
                f"  --seq_len {SEQ_LEN} \\\n"
                f"  --pred_len 0 \\\n"
                f"  --e_layers 2 \\\n"
                f"  --d_layers 1 \\\n"
                f"  --enc_in {enc_in} \\\n"
                f"  --dec_in {enc_in} \\\n"
                f"  --c_out 1 \\\n"             # 类别数 (假设是2类)
                f"  --d_model 64 \\\n"
                f"  --d_ff 64 \\\n"
                f"  --top_k 3 \\\n"
                f"  --des 'Auto_Run' \\\n"
                f"  --itr 1 \\\n"
                f"  --batch_size {BATCH_SIZE} \\\n"
                f"  --learning_rate {LEARNING_RATE} \\\n"
                f"  --train_epochs {EPOCHS} \\\n"
                f"  --patience 3"
            )
            
            # 模型特定参数微调
            if model == 'PatchTST':
                cmd += " --patch_len 16 --stride 8"
            elif model == 'Informer':
                cmd += " --factor 1"
                
            # 执行命令
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"  [Error] {model} failed on {device_name}")
            else:
                print(f"  [Success] {model} finished on {device_name}")

if __name__ == "__main__":
    main()