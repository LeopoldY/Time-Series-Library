import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= ⚙️ 配置区域 =================
# 输入：原始 CSV 文件夹
INPUT_ROOT = "dataset/fault_selected_cleaned/"
# 输出：UEA 数据集根目录
OUTPUT_ROOT = "dataset/fault_selected_for_classification/"
# 输出的数据集名称
DATASET_NAME = "Fault_All"

# 目标列
TARGET_COL = 'w_level4' 
# 窗口长度
SEQ_LEN = 12 
# 滑动步长 (设为1以获取最大样本量)
STRIDE = 3 
# ===============================================

def write_ts_file(X_data, y_data, filename, dataset_name):
    """写出标准 UEA .ts 文件"""
    if not X_data: return
    
    n_samples = len(X_data)
    n_dims = X_data[0].shape[0] # (Dims, Length)
    ts_len = X_data[0].shape[1]
    
    print(f"    -> Writing {os.path.basename(filename)}...")
    print(f"       Samples: {n_samples} | Dims: {n_dims} | Length: {ts_len}")

    with open(filename, 'w') as f:
        f.write(f"@problemName {dataset_name}\n")
        f.write(f"@timeStamps false\n")
        f.write(f"@missing false\n")
        f.write(f"@univariate {str(n_dims==1).lower()}\n")
        f.write(f"@dimensions {n_dims}\n")
        f.write(f"@equalLength true\n")
        f.write(f"@seriesLength {ts_len}\n")
        f.write(f"@classLabel true 0 1\n") 
        f.write(f"@data\n")

        for i in range(n_samples):
            series_str = []
            for dim in range(n_dims):
                dim_val = ",".join(map(str, X_data[i][dim]))
                series_str.append(dim_val)
            
            line = ":".join(series_str) + ":" + str(int(y_data[i]))
            f.write(line + "\n")

def process_single_csv(csv_path):
    """处理单个 CSV 文件，返回切片后的 X 和 y 列表"""
    device_name = os.path.basename(csv_path)
    # print(f"  Reading {device_name}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read {device_name}: {e}")
        return [], []

    # 1. 时间特征提取
    if 'date' not in df.columns:
        return [], []
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['date'].dt.hour
    
    # 移除非数值列
    df_numeric = df.drop(columns=['date'])
    
    if TARGET_COL not in df_numeric.columns:
        return [], []

    # 2. 归一化 (StandardScaler)
    # 注意：我们在切片前对整个设备的数据做归一化，
    # 这样能保留该设备内部的时间趋势，同时消除不同设备间的量级差异
    data_values = df_numeric.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # 目标列的原始值用于生成标签
    raw_target = df[TARGET_COL].values
    
    X_list = []
    y_list = []
    
    total_len = len(data_scaled)
    
    # 3. 滑动窗口切片
    for i in range(0, total_len - SEQ_LEN, STRIDE):
        # 预测目标：窗口结束后的下一个小时
        if i + SEQ_LEN < total_len:
            # Input: (SEQ_LEN, Dims)
            window = data_scaled[i : i + SEQ_LEN]
            
            # Label: 下一时刻 w_level4 > 0
            next_val = raw_target[i + SEQ_LEN]
            label = 1 if next_val > 0 else 0
            
            # Transpose to (Dims, SEQ_LEN)
            X_list.append(window.T)
            y_list.append(label)
            
    return X_list, y_list

def main():
    print(f"Start merging CSVs from {INPUT_ROOT}...")
    
    all_files = glob.glob(os.path.join(INPUT_ROOT, "*.csv"))
    if not all_files:
        print("No CSV files found.")
        return

    # 收集所有数据
    full_X = []
    full_y = []
    
    for f in all_files:
        X, y = process_single_csv(f)
        full_X.extend(X)
        full_y.extend(y)
        print(f"  + {os.path.basename(f)}: Added {len(X)} samples.")
        
    print(f"\nTotal Samples Collected: {len(full_X)}")
    
    if len(full_X) == 0:
        print("Error: No samples generated.")
        return

    # 4. 打乱并划分 (Shuffle & Split)
    # random_state=42 保证可复现
    print("Shuffling and splitting (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        full_X, full_y, test_size=0.2, shuffle=True, random_state=42
    )

    # 统计正负样本数量
    total_pos = int(sum(full_y))
    total_neg = len(full_y) - total_pos
    train_pos = int(sum(y_train))
    train_neg = len(y_train) - train_pos
    test_pos = int(sum(y_test))
    test_neg = len(y_test) - test_pos

    # 5. 保存到 Fault_All 文件夹
    save_dir = os.path.join(OUTPUT_ROOT, DATASET_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    write_ts_file(X_train, y_train, os.path.join(save_dir, f"{DATASET_NAME}_TRAIN.ts"), DATASET_NAME)
    write_ts_file(X_test, y_test, os.path.join(save_dir, f"{DATASET_NAME}_TEST.ts"), DATASET_NAME)
    
    print("\n[Done] Unified dataset created at:")
    print(f"       {save_dir}")

    # 最终输出正负样本数量
    print("\nSample counts:")
    print(f"  Total -> Positive: {total_pos} | Negative: {total_neg}")
    print(f"  Train -> Positive: {train_pos} | Negative: {train_neg}")
    print(f"  Test  -> Positive: {test_pos} | Negative: {test_neg}")

if __name__ == "__main__":
    main()