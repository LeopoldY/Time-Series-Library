import pandas as pd
import os
import glob
import shutil

# ================= 配置区域 =================
# 1. 原始数据根目录 (包含多级子文件夹)
# 请修改为你存放原始数据的路径
SOURCE_DIR = r"dataset/fault/raw_data/"

# 2. 清洗后数据保存目录 (所有文件将平铺保存在这里)
TARGET_DIR = r"dataset/fault/raw_data_cleaned/"

# ===========================================

def clean_and_save(file_path, target_dir):
    try:
        # 1. 读取数据 (尝试多种编码)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                print(f"[Error] 无法解码文件: {file_path}")
                return False

        # 记录原始列数
        original_cols = len(df.columns)
        
        # 2. 核心清洗逻辑：删除所有值均为 NaN 或空字符串的列
        # 将仅包含空格的字符串替换为 NaN
        df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        # 删除全为空的列 (axis=1 表示列, how='all' 表示全空才删)
        df_cleaned = df.dropna(axis=1, how='all')
        
        cleaned_cols = len(df_cleaned.columns)
        dropped_count = original_cols - cleaned_cols

        # 3. 构造新的文件名 (使用设备名以防冲突)
        # 假设路径结构: .../[Category]/[Device_Name]/xyz.csv
        # 我们取文件所在的父文件夹名作为设备名
        device_name = os.path.basename(file_path).split('_')[0]
        new_filename = f"{device_name}_For_Machine_Learning.csv"
        
            
        target_path = os.path.join(target_dir, new_filename)

        # 4. 保存文件
        df_cleaned.to_csv(target_path, index=False, encoding='utf-8')
        
        print(f"[Success] {device_name}: 删除了 {dropped_count} 个空列. 保存至 -> {new_filename}")
        return True

    except Exception as e:
        print(f"[Failed] 处理 {file_path} 时出错: {e}")
        return False

def main():
    print("="*50)
    print("开始数据清洗与扁平化处理")
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print("="*50)

    # 1. 创建目标目录
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"已创建目标目录: {TARGET_DIR}")

    # 2. 递归查找所有 CSV 文件
    # 使用 glob 的 recursive=True 参数
    search_pattern = os.path.join(SOURCE_DIR, "*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print("未找到任何 CSV 文件，请检查路径。")
        return

    print(f"共发现 {len(files)} 个原始文件，开始处理...\n")

    success_count = 0
    for i, file_path in enumerate(files):
        if clean_and_save(file_path, TARGET_DIR):
            success_count += 1

    print("\n" + "="*50)
    print("处理完成!")
    print(f"成功处理并保存: {success_count} / {len(files)}")
    print(f"清洗后的数据位于: {TARGET_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()