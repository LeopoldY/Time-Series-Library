import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import shutil

# ================= 配置区域 =================
DATASET_ROOT = "dataset/fault_raw/processed_rich" # 或 raw_data_cleaned 对应的处理后目录
OUTPUT_NPY_DIR = "/mnt/sdc1/skx/Time-Series-Library/results_xgboost/" # 专门存放 XGBoost 原始结果
SEQ_LENS = [3, 6, 12, 24]
PRED_LEN = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# ===========================================

def create_sliding_window(data, seq_len, pred_len, target_idx):
    Xs, ys = [], []
    n_samples = len(data) - seq_len - pred_len + 1
    for i in range(n_samples):
        Xs.append(data[i : i + seq_len].flatten())
        ys.append(data[i + seq_len : i + seq_len + pred_len, target_idx])
    return np.array(Xs), np.array(ys).flatten()

def main():
    if os.path.exists(OUTPUT_NPY_DIR):
        shutil.rmtree(OUTPUT_NPY_DIR)
    os.makedirs(OUTPUT_NPY_DIR)
    
    files = glob.glob(os.path.join(DATASET_ROOT, "*.csv"))
    print(f"Found {len(files)} devices for XGBoost training...")

    for file_path in files:
        device_name = os.path.basename(file_path).replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            # 假设最后一列是 date，我们要把它排除，且假设 w_level4 是预测目标
            if 'date' in df.columns: df = df.drop(columns=['date'])
            
            # 确定 target 列索引 (w_level4)
            cols = list(df.columns)
            if 'w_level4' not in cols: continue
            target_idx = cols.index('w_level4')
            
            data_values = df.values
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_values)

            for seq_len in SEQ_LENS:
                X, y = create_sliding_window(data_scaled, seq_len, PRED_LEN, target_idx)
                if len(X) < 50: continue

                # 数据切分 (Train/Val/Test) - 保持与 Transformer 一致
                n_train = int(len(X) * TRAIN_RATIO)
                n_test = int(len(X) * (1 - TRAIN_RATIO - VAL_RATIO)) # 实际上通常是剩余的为Test
                
                # 这里简化处理，直接取最后 20% 作为测试集用于对比
                split_idx = int(len(X) * 0.8)
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_test, y_test = X[split_idx:], y[split_idx:]

                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # 保存为 .npy (模拟 TSL 的目录结构)
                # 文件夹名格式: XGBoost_{设备名}_{序列长度}
                folder_name = f"XGBoost_{device_name}_{seq_len}"
                save_dir = os.path.join(OUTPUT_NPY_DIR, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # 重点：保存 pred.npy 和 true.npy
                np.save(os.path.join(save_dir, "pred.npy"), preds)
                np.save(os.path.join(save_dir, "true.npy"), y_test)
                
                print(f"Saved: {folder_name}")

        except Exception as e:
            print(f"Error {device_name}: {e}")

if __name__ == "__main__":
    main()