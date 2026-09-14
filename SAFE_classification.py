import os
import glob
import pandas as pd
import numpy as np

# ================= 显卡配置区域 =================
# 指定使用 ID 为 1 的显卡
# 注意：必须在导入 xgboost/torch 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ================= 1. 配置区域 =================
# [输入1] 原始 CSV 路径 (用于计算特征，决定路由)
RAW_CSV_DIR = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault_selected/"

# [输入2] 转换好的 UEA 数据集根目录 (用于训练)
# 结构应该是: .../dataset/UEA/设备83/设备83_TRAIN.ts
UEA_ROOT = "/mnt/sdc1/skx/Time-Series-Library/dataset/UEA/"

# 目标设备 (必须同时存在于 CSV 和 UEA 文件夹中)
TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]

# 原始数据中的目标列 (用于计算稀疏度)
TARGET_COL = 'w_level4'

# 序列长度 (需与转换TS文件时的一致)
SEQ_LEN = 29
# ===============================================

class SafePipeline:
    def __init__(self):
        print("Initializing SAFE Pipeline (Hybrid Mode)...")
        
    def get_ts_dimensions(self, device_name):
        """
        从 .ts 文件头自动读取输入维度 (enc_in)
        这是最准确的方法，避免了 CSV 和 TS 维度不一致的问题
        """
        ts_path = os.path.join(UEA_ROOT, device_name, f"{device_name}_TRAIN.ts")
        if not os.path.exists(ts_path):
            print(f"[Error] TS file not found: {ts_path}")
            return 0
            
        try:
            with open(ts_path, 'r') as f:
                for line in f:
                    if line.startswith("@dimensions"):
                        return int(line.split()[1])
        except:
            return 0
        return 0

    def extract_features(self, csv_path):
        """
        Step 1: 从原始 CSV 提取特征 (Mean, Sparsity, CV)
        """
        try:
            df = pd.read_csv(csv_path)
            if TARGET_COL not in df.columns:
                print(f"[Warn] Target col '{TARGET_COL}' not found in {csv_path}")
                return None
                
            data = df[TARGET_COL].values
            
            # 计算统计特征
            mean_val = np.mean(data)
            # 稀疏度: 0 的占比
            sparsity = (data == 0).sum() / len(data)
            # 变异系数: 标准差 / 均值
            cv_val = np.std(data) / (mean_val + 1e-6)
            
            return {'Sparsity': sparsity, 'CV': cv_val, 'Mean': mean_val}
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            return None

    def route_model(self, features):
        """
        Step 2: 路由逻辑 (Clustering Strategy)
        """
        if features is None: return 'TimesNet', 'Default'
        
        sp = features['Sparsity']
        cv = features['CV']
        
        # --- 路由规则 (可根据您的论文调整) ---
        if sp > 0.90:
            # 极度稀疏 -> PatchTST (擅长长程捕捉)
            return 'PatchTST', f'High Sparsity ({sp:.2f})'
        elif cv > 2.0:
            # 高波动 -> Informer (Attention 关注突发)
            return 'Informer', f'High Volatility (CV={cv:.2f})'
        else:
            # 平稳/周期 -> TimesNet (2D 卷积捕捉周期)
            return 'TimesNet', f'General Pattern'

    def run_training(self, device_name, model_name, enc_in):
        """
        Step 3: 使用 TSLib 标准分类流程进行训练
        """
        print(f"  >>> Launching Training: {model_name} on {device_name} (Dims={enc_in})...")
        
        # 构造标准 UEA 训练命令
        cmd = (
            f"python -u run.py \\\n"
            f"  --task_name classification \\\n"
            f"  --is_training 1 \\\n"
            f"  --root_path {UEA_ROOT}/{device_name} \\\n"     # 指向 UEA 根目录
            f"  --data_path {device_name} \\\n"  # 文件夹名
            f"  --model_id SAFE_{device_name} \\\n"
            f"  --model {model_name} \\\n"       # SAFE 选出的模型
            f"  --data UEA \\\n"                 # 指定数据类型为 UEA
            f"  --seq_len {SEQ_LEN} \\\n"        # 必须与TS文件一致
            f"  --pred_len 0 \\\n"               # 分类任务
            f"  --e_layers 2 \\\n"
            f"  --d_layers 1 \\\n"
            f"  --enc_in {enc_in} \\\n"
            f"  --dec_in {enc_in} \\\n"
            f"  --c_out 2 \\\n"                  # 2分类
            f"  --d_model 64 \\\n"               # 轻量化模型
            f"  --d_ff 64 \\\n"
            f"  --top_k 3 \\\n"
            f"  --des 'SAFE_Exp' \\\n"
            f"  --itr 1 \\\n"
            f"  --batch_size 16 \\\n"
            f"  --learning_rate 0.001 \\\n"
            f"  --train_epochs 20 \\\n"
            f"  --patience 5"
        )
        
        # 模型特定参数
        if model_name == 'PatchTST':
            cmd += " --patch_len 16 --stride 8"
        elif model_name == 'Informer':
            # 防止短序列/分类任务报错
            cmd += " --factor 1 --output_attention"
            
        # 执行
        exit_code = os.system(cmd)
        if exit_code != 0:
            print(f"  [Error] Training failed for {device_name}")
        else:
            print(f"  [Success] Finished.")

    def run(self):
        # 扫描原始 CSV
        csv_files = glob.glob(os.path.join(RAW_CSV_DIR, "*.csv"))
        
        for f in csv_files:
            # 获取设备名 (假设文件名包含设备名)
            fname = os.path.basename(f).replace('.csv', '')
            
            # 过滤非目标设备
            matched_device = None
            for t in TARGET_DEVICES:
                if t in fname:
                    matched_device = t # 使用列表里的标准名称
                    break
            
            if not matched_device: continue
            
            print(f"\n================ Processing: {matched_device} ================")
            
            # 1. 检查 TS 数据是否存在并获取维度
            enc_in = self.get_ts_dimensions(matched_device)
            if enc_in == 0:
                print(f"  [Skip] TS file missing or invalid for {matched_device}")
                continue
            
            # 2. 提取特征 (读取 CSV)
            feats = self.extract_features(f)
            if not feats: continue
            print(f"  [Feature] Sparsity: {feats['Sparsity']:.3f} | CV: {feats['CV']:.3f}")
            
            # 3. 路由选择
            model, reason = self.route_model(feats)
            print(f"  [Router] Selected Model: {model}")
            print(f"           Reason: {reason}")
            
            # 4. 执行训练 (读取 TS)
            self.run_training(matched_device, model, enc_in)

if __name__ == "__main__":
    pipeline = SafePipeline()
    pipeline.run()
