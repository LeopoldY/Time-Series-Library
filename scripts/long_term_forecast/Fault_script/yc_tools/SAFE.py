import os
import sys
import pandas as pd
import glob
import numpy as np

from tqdm import tqdm

# 添加当前路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import run_tools as utils
import cluster as models

# ================= 配置区域 =================
# 1. 原始清洗后的数据路径 (你的输入)
RAW_CLEANED_DIR = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault_raw/raw_data_cleaned"

# 2. 特征工程后的数据保存路径 (模型实际读取这里)
PROCESSED_DIR = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault_raw/processed_rich/"

# 3. 脚本输出路径
OUTPUT_SCRIPT_DIR = "/mnt/sdc1/skx/Time-Series-Library/scripts/long_term_forecast/Fault_script/ada_run/"

# 4. 聚类结果图片输出路径
CLUSTER_PLOT_DIR = "/mnt/sdc1/skx/Time-Series-Library/visualization/cluster_plots/"

# 5. 特征工程参数
TIME_WINDOW = '1H'
TOP_N_TYPES = 10  # 只保留前10种最频繁的告警类型，其余归为 Other

# 6. 策略映射
STRATEGY_MAPPING = {
    'Dense': 'Informer',
    'Sparse': 'PatchTST',
    'Intermittent': 'iTransformer'
}
# ===========================================

def convert_raw_data(input_dir, output_dir):
    """
    核心新函数：将原始流水账数据转换为多维时间序列特征。
    返回: 转换后的特征维度 (enc_in)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"[Error] No CSV files found in {input_dir}")
        return 0

    print(f"\n[1/4] Feature Engineering on {len(csv_files)} files...")
    
    max_features = 0
    processed_count = 0

    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            # 读取
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                df = pd.read_csv(file_path, encoding='gbk')

            # 必须包含时间列
            if '发生时间' not in df.columns:
                print(f"Skip {os.path.basename(file_path)}: No '发生时间' column")
                continue

            # 1. 时间索引
            df['发生时间'] = pd.to_datetime(df['发生时间'], errors='coerce')
            df = df.dropna(subset=['发生时间'])
            df.set_index('发生时间', inplace=True)

            # 2. 基础特征: 级别 (Mapping)
            # 兼容中文 '紧急' 和数字 4
            level_map = {
                '提示': 'w_level1', 1: 'w_level1', '1': 'w_level1',
                '次要': 'w_level2', 2: 'w_level2', '2': 'w_level2',
                '重要': 'w_level3', 3: 'w_level3', '3': 'w_level3',
                '紧急': 'w_level4', 4: 'w_level4', '4': 'w_level4'
            }
            if '级别' in df.columns:
                df['level_mapped'] = df['级别'].map(level_map)
                df_levels = pd.get_dummies(df['level_mapped']).resample(TIME_WINDOW).sum()
            else:
                # 如果没有级别列，生成全0
                df_levels = pd.DataFrame()

            # 补全 w_level1-4
            for col in ['w_level1', 'w_level2', 'w_level3', 'w_level4']:
                if col not in df_levels.columns:
                    df_levels[col] = 0
            
            # 3. 高级特征: 告警类型 (Top N)
            if '告警类型' in df.columns:
                top_types = df['告警类型'].value_counts().nlargest(TOP_N_TYPES).index
                df['type_filtered'] = df['告警类型'].apply(lambda x: x if x in top_types else 'Other')
                df_types = pd.get_dummies(df['type_filtered'], prefix='type').resample(TIME_WINDOW).sum()
            else:
                df_types = pd.DataFrame()

            # 4. 时效特征: 修复时长
            if 'Repair_Interval' in df.columns:
                df['Repair_Interval'] = pd.to_numeric(df['Repair_Interval'], errors='coerce')
                df_repair = df['Repair_Interval'].resample(TIME_WINDOW).mean().fillna(0)
                df_repair.name = 'avg_repair_time'
            else:
                df_repair = pd.Series(0, index=df_levels.index, name='avg_repair_time')

            # 5. 合并
            df_final = pd.concat([df_levels, df_types, df_repair], axis=1).fillna(0)
            
            # 格式化: 必须把 date 放回列中
            df_final.reset_index(inplace=True)
            df_final.rename(columns={'发生时间': 'date'}, inplace=True)

            # 保存
            file_name = os.path.basename(file_path)
            save_path = os.path.join(output_dir, file_name)
            df_final.to_csv(save_path, index=False)
            
            # 更新特征数 (列数 - 1个date列)
            current_feat_num = df_final.shape[1] - 1
            if current_feat_num > max_features:
                max_features = current_feat_num
            
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Processed {processed_count} files. Max features detected: {max_features}")
    return max_features

def main():
    print("="*50)
    print("SAFE Framework: Raw Data Pipeline")
    print("="*50)
    
    # 1. 数据转化 (新增步骤)
    # 这会将 raw_data_cleaned -> processed_rich
    # 并返回特征数量 (例如 15)
    enc_in = convert_raw_data(RAW_CLEANED_DIR, PROCESSED_DIR)
    
    if enc_in == 0:
        print("[Error] Feature engineering failed. Exiting.")
        return
        
    print(f"\n[Info] Determined enc_in = {enc_in}")

    # 2. 查找处理后的文件
    # 注意：models.py 需要的是 (path, name, category)
    # 我们这里简化处理，直接扫描 processed_rich 目录
    csv_files = []
    for f in glob.glob(os.path.join(PROCESSED_DIR, "*.csv")):
        name = os.path.basename(f).replace('.csv', '')
        csv_files.append((f, name, 'Unknown_Cat')) # 类别在这里不影响聚类

    # 3. 特征提取 (Sparsity/CV)
    print(f"\n[2/4] Profiling devices based on w_level4...")
    features = []
    for path, name, cat in tqdm(csv_files, desc="Extracting features"):
        # extract_features_from_file 只看 w_level4，这在新数据里依然存在
        feat = models.extract_features_from_file(path, name, cat)
        if feat: features.append(feat)
    
    df_feat = pd.DataFrame(features)
    print(f"Valid devices for training: {len(df_feat)}")

    # 4. 聚类
    print("\n[3/4] Performing Clustering...")
    df_clustered = models.perform_clustering(df_feat, plot_dir=CLUSTER_PLOT_DIR)
    print(df_clustered['Label'].value_counts())

    

    # 5. 生成脚本 (调用 utils 中新添加的函数)
    print(f"\n[4/4] Generating scripts in {OUTPUT_SCRIPT_DIR}...")
    utils.generate_scripts_dynamic(
        df_clustered, 
        OUTPUT_SCRIPT_DIR, 
        STRATEGY_MAPPING,
        enc_in=enc_in  # <--- 传入计算出的特征维度
    )

    print("="*50)
    print("Pipeline Completed.")
    print("Run 'python scripts/long_term_forecast/Fault_script/yc_tools/run_all_ada.py' to start.")
    print("="*50)

if __name__ == "__main__":
    main()