import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    print("Running Standard UEA Classification...")
    
    # 自动获取 Channel 数 (enc_in)
    # 读取 .ts 文件的头部信息
    ts_path = f"{ROOT_PATH}/{DATA_ID}_TRAIN.ts"
    enc_in = 0
    with open(ts_path, 'r') as f:
        for line in f:
            if line.startswith("@dimensions"):
                enc_in = int(line.split()[1])
                break
    print(f"Detected Dimensions (enc_in): {enc_in}")

    for model in MODELS:
        print(f"\n>>> Training {model} ...")
        
        # 标准分类命令
        cmd = (
            f"python -u run.py \\\n"
            f"  --task_name classification \\\n"  # 使用标准分类任务
            f"  --is_training 1 \\\n"
            f"  --root_path {ROOT_PATH} \\\n" # 指向数据集路径
            f"  --model_id {DATA_ID}_{model} \\\n"
            f"  --model {model} \\\n"
            f"  --data UEA \\\n"                 # 数据类型设为 UEA
            f"  --data_path {DATA_ID} \\\n"      # 数据集名称
            f"  --seq_len {SEQ_LEN} \\\n"        # 必须与 .ts 文件里的长度一致
            f"  --pred_len 0 \\\n"               # 分类任务 pred_len 为 0
            f"  --e_layers 2 \\\n"
            f"  --d_layers 1 \\\n"
            f"  --factor 3 \\\n"
            f"  --enc_in {enc_in} \\\n"
            f"  --dec_in {enc_in} \\\n"
            f"  --c_out 2 \\\n"                  # 类别数 (0/1 两类)
            f"  --d_model 64 \\\n"               # 分类任务通常模型可以小一点
            f"  --d_ff 64 \\\n"
            f"  --top_k 3 \\\n"                  # TimesNet 参数
            f"  --des 'Auto_Run' \\\n"
            f"  --itr 1 \\\n"
            f"  --batch_size {BATCH_SIZE} \\\n"
            f"  --learning_rate {LEARNING_RATE} \\\n"
            f"  --train_epochs {EPOCHS} \\\n"
            f"  --patience 3"
        )
        
        # PatchTST 特定参数
        if model == 'PatchTST':
            cmd += " --patch_len 16 --stride 8"
        elif model == 'Nonstationary_Transformer':
            cmd += "  --p_hidden_dims 256 256 --p_hidden_layers 2 "

        exit_code = os.system(cmd)
        if exit_code != 0:
            print(f"Error running {model}")

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Classification')
    args.add_argument('--squence_length', type=int, default=96, help='input sequence length of Informer encoder')
    args = args.parse_args()
    # ================= 配置 =================
    SEQ_LEN = args.squence_length
    # 这里的名字必须和 dataset/UEA/ 下的文件夹名字一致
    DATA_ID = "Fault_All{}".format(SEQ_LEN) 
    ROOT_PATH = "dataset/fault_selected_for_classification/Fault_All{}".format(SEQ_LEN)
    # 我们的滑动窗口长度
    

    # 推荐模型列表
    MODELS = [    
        'Nonstationary_Transformer',   
        'TimesNet',          # 分类任务目前的最强基线 (ICLR 2023)
        'Informer',          # 您之前的基线
    ]

    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 30
    # =======================================
    main()