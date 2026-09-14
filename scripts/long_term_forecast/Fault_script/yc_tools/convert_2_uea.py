import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= ⚙️ 配置区域 =================
# 历史窗口长度
SEQ_LEN = 24 
# 原始 CSV 文件夹路径
INPUT_DIR = "dataset/fault_selected/"
# 输出 UEA 格式路径 (生成 Fault_All 文件夹)
OUTPUT_DIR = "dataset/fault_selected_for_classification/"
# 输出数据集名称
DATASET_NAME = f"Fault_All{SEQ_LEN}"

# 目标列 (用于生成标签)
TARGET_COL = 'w_level4' 



# 【核心策略】自适应步长设置
STRIDE_FAULT = 1   # 故障样本：每1个点采一次 (精细)
STRIDE_NORMAL = int(SEQ_LEN / 4)  # 正常样本：每4个点采一次 (降采样，减少负样本)
# ===============================================

def write_ts_file(X_data, y_data, filename, dataset_name):
    """写出符合 UEA 标准的 .ts 文件"""
    if not X_data: 
        print(f"Warning: No data for {filename}")
        return

    n_samples = len(X_data)
    n_dims = X_data[0].shape[0] # (Dims, Length)
    ts_len = X_data[0].shape[1]
    
    print(f"    -> Writing {os.path.basename(filename)}...")
    print(f"       Samples: {n_samples} | Dims: {n_dims} (Data+Time) | Length: {ts_len}")

    with open(filename, 'w') as f:
        f.write(f"@problemName {dataset_name}\n")
        f.write(f"@timeStamps false\n") # 时间已转化为特征，不需要额外的时间戳列
        f.write(f"@missing false\n")
        f.write(f"@univariate false\n")
        f.write(f"@dimensions {n_dims}\n")
        f.write(f"@equalLength true\n")
        f.write(f"@seriesLength {ts_len}\n")
        f.write(f"@classLabel true 0 1\n") # 0:正常, 1:故障
        f.write(f"@data\n")

        for i in range(n_samples):
            # X_data[i] shape: (Dims, Seq_Len)
            series_str = []
            for dim in range(n_dims):
                # 将该维度的时间序列转为逗号分隔字符串
                dim_val = ",".join(map(str, X_data[i][dim]))
                series_str.append(dim_val)
            
            # 拼接: dim1:dim2:...:label
            line = ":".join(series_str) + ":" + str(int(y_data[i]))
            f.write(line + "\n")

def process_single_csv(csv_path):
    """处理单个文件：提取时间特征 + 归一化 + 自适应切片"""
    device_name = os.path.basename(csv_path)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read {device_name}: {e}")
        return [], []

    # 1. 时间特征提取 (Time Embedding)
    if 'date' not in df.columns:
        print(f"  [Skip] No 'date' column in {device_name}")
        return [], []
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['date'].dt.hour
    
    # 移除原始 date 对象，保留数值列
    # 此时 df_numeric 包含: [w_level1~4, month, day, weekday, hour]
    df_numeric = df.drop(columns=['date'])
    
    if TARGET_COL not in df_numeric.columns:
        return [], []

    # 2. 归一化 (StandardScaler)
    # 对所有列(包括时间)进行归一化，这对 Transformer/TimesNet 非常重要
    data_values = df_numeric.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # 保留原始的目标列用于判断标签 (因为归一化后就看不出是否 > 0 了)
    raw_target = df[TARGET_COL].values
    
    X_list = []
    y_list = []
    
    total_len = len(data_scaled)
    i = 0
    
    # 3. 自适应滑动窗口 (Adaptive Stride)
    while i < total_len - SEQ_LEN:
        # 预测目标索引: 窗口结束后的下一个点
        target_idx = i + SEQ_LEN
        
        if target_idx >= total_len:
            break
            
        # 标签逻辑: 下一个小时 w_level4 > 0 则为故障(1)
        is_fault = (raw_target[target_idx] > 0)
        label = 1 if is_fault else 0
        
        # 提取窗口 (Seq_Len, Dims)
        window = data_scaled[i : i + SEQ_LEN]
        
        # 添加数据
        X_list.append(window.T) # 转置为 (Dims, Seq_Len)
        y_list.append(label)
        
        # === 核心：动态调整步长 ===
        if is_fault:
            # 故障样本：高频采样
            stride = STRIDE_FAULT 
            
            # 【可选】数据增强：如果样本极少，可以在这里复制一份带噪声的样本
            # noise = np.random.normal(0, 0.05, window.shape)
            # X_list.append((window + noise).T)
            # y_list.append(label)
        else:
            # 正常样本：低频采样 (解决不平衡)
            stride = STRIDE_NORMAL
            
            # 特判：如果未来12小时内有故障，不要跳过，保持密集采样捕捉前兆
            future_check = raw_target[target_idx : min(target_idx+12, total_len)]
            if np.sum(future_check) > 0:
                stride = STRIDE_FAULT

        i += stride
            
    return X_list, y_list

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Start processing CSVs from {INPUT_DIR}...")
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not all_files:
        print("No CSV files found.")
        return

    # 收集所有设备的数据
    full_X = []
    full_y = []
    
    for f in all_files:
        print(f"Processing {os.path.basename(f)}...")
        X, y = process_single_csv(f)
        full_X.extend(X)
        full_y.extend(y)
        
    print(f"\nTotal Samples Collected: {len(full_X)}")
    
    # 统计正负样本比例
    pos_count = sum(full_y)
    neg_count = len(full_y) - pos_count
    print(f"Positive (Fault): {pos_count}")
    print(f"Negative (Normal): {neg_count}")
    if pos_count > 0:
        print(f"Ratio: 1 : {neg_count/pos_count:.2f}")

    if len(full_X) == 0:
        print("Error: No samples generated.")
        return

    # 4. 打乱并划分 (Shuffle & Split)
    # 混合所有设备数据后，随机切分，保证训练集和测试集分布一致
    print("Shuffling and splitting (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        full_X, full_y, test_size=0.2, shuffle=True, random_state=42
    )

    # 5. 保存结果
    save_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    write_ts_file(X_train, y_train, os.path.join(save_dir, f"{DATASET_NAME}_TRAIN.ts"), DATASET_NAME)
    write_ts_file(X_test, y_test, os.path.join(save_dir, f"{DATASET_NAME}_TEST.ts"), DATASET_NAME)
    
    print(f"\n[Done] Unified dataset created at: {save_dir}")
    print(f"       Use '--enc_in {X_train[0].shape[0]}' in your training script.")

if __name__ == "__main__":
    main()