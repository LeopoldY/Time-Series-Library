import os

# ================= 显卡配置区域 =================
# 指定使用 ID 为 1 的显卡
# 注意：必须在导入 xgboost/torch 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ===============================================

import pandas as pd
import numpy as np
import glob
import shutil
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 1. 输入数据路径
DATA_DIR = "dataset/fault_selected/"

# 2. 输出配置
OUTPUT_CSV = "stacking_results_4_devices.csv" 
OUTPUT_NPY_DIR = "/mnt/sdc1/skx/Time-Series-Library/results_stacking/"

# 3. 实验对象
TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]
SEQ_LENS = [3, 6, 12, 24]
PRED_LEN = 1
TARGET_COL = 'w_level4'
# ===========================================

def create_dataset(data, seq_len, pred_len, target_idx):
    Xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        Xs.append(data[i:i+seq_len].flatten())
        ys.append(data[i+seq_len:i+seq_len+pred_len, target_idx])
    return np.array(Xs), np.array(ys).flatten()

def main():
    print("="*50)
    print("Running Stacking Complete Experiment (GPU Mode: Device 1)")
    print("Tasks: 1. Save MSE/MAE to CSV  2. Save NPY for Classification")
    print("="*50)
    
    # 初始化输出目录
    if os.path.exists(OUTPUT_NPY_DIR):
        try:
            shutil.rmtree(OUTPUT_NPY_DIR)
        except Exception as e:
            print(f"[Warn] Could not delete directory: {e}")
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

    # 筛选文件
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    target_files = []
    for f in all_files:
        fname = os.path.basename(f)
        for t in TARGET_DEVICES:
            if t in fname:
                target_files.append(f)
                break
    
    print(f"Found {len(target_files)} target files.")
    
    csv_records = []

    for file_path in target_files:
        device_name = os.path.basename(file_path).replace('.csv', '')
        print(f"\nProcessing: {device_name}")
        
        try:
            # 读取与预处理
            df = pd.read_csv(file_path)
            if 'date' in df.columns: df = df.drop(columns=['date'])
            if TARGET_COL not in df.columns: 
                print(f"  [Skip] Missing target column in {device_name}")
                continue
                
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df.values)
            target_idx = list(df.columns).index(TARGET_COL)
            
            for seq_len in SEQ_LENS:
                X, y = create_dataset(data_scaled, seq_len, PRED_LEN, target_idx)
                
                if len(X) < 50: 
                    print(f"  [Skip] Not enough data for seq_len={seq_len}")
                    continue
                
                # 划分训练/测试集
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # === 训练 Stacking (启用 GPU) ===
                estimators = [
                    ('xgb', xgb.XGBRegressor(
                        n_estimators=50, 
                        max_depth=5, 
                        random_state=42,
                        # 【关键修改】启用 GPU 加速
                        tree_method='hist', 
                        device='cuda',
                        # 注意：在使用 GPU 时，建议减少 XGB 内部的 n_jobs，交给 GPU 处理并行
                        n_jobs=1 
                    )),
                    ('rf', RandomForestRegressor(
                        n_estimators=50, 
                        max_depth=5, 
                        n_jobs=4, # 随机森林还是跑在 CPU 上，可以用多核
                        random_state=42
                    ))
                ]
                
                # 注意：StackingRegressor 的 n_jobs 如果大于 1，会启动多进程
                # 多进程下初始化 CUDA 可能会报错。如果再次报错，请将此处的 n_jobs 改为 1
                reg = StackingRegressor(
                    estimators=estimators, 
                    final_estimator=LinearRegression(), 
                    n_jobs=1 # 为安全起见，Stacking 层设为 1，防止多进程 CUDA 冲突
                )
                
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                
                # === 记录回归指标 ===
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                csv_records.append({
                    'Device': device_name,
                    'Model': 'Stacking', 
                    'Seq_Len': seq_len,
                    'MSE': mse,
                    'MAE': mae
                })
                print(f"  -> Seq {seq_len}: MSE={mse:.6f}, MAE={mae:.6f}")

                # === 保存 NPY ===
                folder_name = f"Stacking_{device_name}_{seq_len}"
                save_path = os.path.join(OUTPUT_NPY_DIR, folder_name)
                os.makedirs(save_path, exist_ok=True)
                
                np.save(os.path.join(save_path, "pred.npy"), preds)
                np.save(os.path.join(save_path, "true.npy"), y_test)
                
        except Exception as e:
            print(f"Error processing {device_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存 CSV
    if csv_records:
        df_res = pd.DataFrame(csv_records)
        df_res = df_res[['Device', 'Model', 'Seq_Len', 'MSE', 'MAE']]
        df_res.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*50)
        print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")
        print("="*50)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()