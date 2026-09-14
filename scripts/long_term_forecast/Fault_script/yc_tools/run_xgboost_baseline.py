import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time

# ================= 配置区域 =================
# 数据集根目录
DATASET_ROOT = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault/"

# 结果保存路径
OUTPUT_CSV = "xgboost_summary.csv"

# 实验序列长度 (与 Transformer 实验保持一致)
SEQ_LENS = [3, 6, 12, 24]

# 预测步长 (固定为1)
PRED_LEN = 1

# 数据集切分比例 (Train/Test) - 保持与 Time-Series-Library 默认大致一致 (通常是7:1:2或8:2)
TRAIN_RATIO = 0.8 
# ===========================================

def create_sliding_window_dataset(data, seq_len, pred_len, target_col_idx):
    """
    将时间序列转换为 XGBoost 可用的 (X, y) 监督学习格式
    X shape: [Samples, Features * seq_len]
    y shape: [Samples, 1]
    """
    Xs, ys = [], []
    n_samples = len(data) - seq_len - pred_len + 1
    
    for i in range(n_samples):
        # 提取窗口内的所有特征作为输入
        # Flatten: 将 (seq_len, 4) 展平为 (seq_len * 4, )
        x_window = data[i : i + seq_len].flatten()
        
        # 提取预测目标的未来值
        y_val = data[i + seq_len : i + seq_len + pred_len, target_col_idx]
        
        Xs.append(x_window)
        ys.append(y_val)
        
    return np.array(Xs), np.array(ys).flatten()

def find_csv_files(dataset_root):
    csv_files = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".csv"):
                # 获取设备名 (取文件夹名或文件名)
                path_parts = root.strip(os.sep).split(os.sep)
                if len(path_parts) >= 1:
                    device_name = path_parts[-1] # 使用所在文件夹名作为设备名
                else:
                    device_name = file.replace('.csv', '')
                
                full_path = os.path.join(root, file)
                csv_files.append((full_path, device_name))
    return csv_files

def main():
    print("="*50)
    print("Starting XGBoost Baseline Experiment")
    print("="*50)

    # 1. 寻找文件
    files = find_csv_files(DATASET_ROOT)
    if not files:
        print(f"[Error] No CSV files found in {DATASET_ROOT}")
        return
    print(f"Found {len(files)} devices.")

    results = []
    
    # 2. 遍历所有设备
    for idx, (file_path, device_name) in enumerate(files):
        print(f"\n[{idx+1}/{len(files)}] Processing {device_name} ...")
        
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 确保包含所需的特征列
            feature_cols = ['w_level1', 'w_level2', 'w_level3', 'w_level4']
            if not all(col in df.columns for col in feature_cols):
                print(f"  -> Skip: Missing columns in {device_name}")
                continue
                
            # 转换为 Numpy 数组
            data_values = df[feature_cols].values
            target_idx = feature_cols.index('w_level4') # 预测目标列索引
            
            # 数据标准化 (对 XGBoost 不是必须的，但对某些分布有帮助，保持与 DL 一致性)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_values)
            
            # 3. 遍历不同的序列长度
            for seq_len in SEQ_LENS:
                # 构建数据集
                X, y = create_sliding_window_dataset(data_scaled, seq_len, PRED_LEN, target_idx)
                
                if len(X) < 10:
                    print(f"  -> Skip: Not enough data for seq_len={seq_len}")
                    continue
                
                # 划分训练集/测试集 (按时间顺序划分，严禁 Shuffle)
                train_size = int(len(X) * TRAIN_RATIO)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # 初始化 XGBoost 回归器
                # 参数说明: 
                # n_estimators: 树的数量
                # learning_rate: 学习率
                # max_depth: 树深，防止过拟合
                model = xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    objective='reg:squarederror',
                    n_jobs=-1,  # 并行加速
                    random_state=42
                )
                
                # 训练
                model.fit(X_train, y_train)
                
                # 预测
                preds = model.predict(X_test)
                
                # 反归一化 (Inverse Transform) 才能计算真实的 MSE/MAE
                # 注意：我们只缩放了输入，输出也是缩放后的。
                # 简单做法：手动反归一化 y
                # 获取 w_level4 的均值和方差
                mean_y = scaler.mean_[target_idx]
                std_y = scaler.scale_[target_idx]
                
                preds_rescaled = preds * std_y + mean_y
                y_test_rescaled = y_test * std_y + mean_y
                
                # 确保非负 (故障数不能为负)
                preds_rescaled = np.maximum(preds_rescaled, 0)
                
                # 计算指标
                mse = mean_squared_error(y_test_rescaled, preds_rescaled)
                mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
                
                print(f"  -> Seq {seq_len}: MSE={mse:.6f}, MAE={mae:.6f}")
                
                results.append({
                    'Device': device_name,
                    'Seq_Len': seq_len,
                    'MSE': mse,
                    'MAE': mae,
                    'Model': 'XGBoost' # 标记模型名
                })
                
        except Exception as e:
            print(f"  -> Error: {e}")

    # 4. 保存结果
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*50)
        print(f"Done! Results saved to {OUTPUT_CSV}")
        
        # 打印简单摘要
        print("\nXGBoost Performance Summary (Avg MSE):")
        print(res_df.groupby('Seq_Len')['MSE'].mean())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()