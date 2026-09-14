import pandas as pd
import os
import glob

# ================= 配置 =================
# 之前的 rich features 数据路径
INPUT_DIR = "dataset/fault_selected/"
# 新的分类数据保存路径
OUTPUT_DIR = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault_classification/"

TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]
TARGET_COL = 'w_level4'
# =======================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    print(f"Preparing Classification Data for {len(TARGET_DEVICES)} devices...")
    
    for f in all_files:
        # 检查是否是目标设备
        fname = os.path.basename(f)
        is_target = any(d in fname for d in TARGET_DEVICES)
        if not is_target: continue
        
        df = pd.read_csv(f)
        
        # 1. 核心转换：回归 -> 分类
        # 将大于0的值设为1，等于0的保持0
        raw_count = df[TARGET_COL].sum()
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
        new_count = df[TARGET_COL].sum()
        
        # 2. 计算正负样本比例 (用于后续加权)
        total = len(df)
        pos_ratio = new_count / total * 100
        print(f"[{fname}] Faults: {raw_count} -> {new_count} (Pos Rate: {pos_ratio:.2f}%)")
        
        # 3. 保存
        df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    print(f"\nData prepared in {OUTPUT_DIR}")
    print("Use this path as your new --root_path")

if __name__ == "__main__":
    main()