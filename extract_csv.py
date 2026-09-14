import os
import numpy as np
import pandas as pd
import re
import glob

# ================= 配置区域 =================
# 请根据您服务器的实际路径修改
# 1. SAFE-Basic 实验结果目录
BASIC_RESULTS_DIR = "/mnt/sdc1/skx/Time-Series-Library/results_SAFE-Basic/"
# BASIC 对应的输出文件名
BASIC_OUTPUT_CSV = "SAFE_Basic_results.csv"

# 2. SAFE-Rich 实验结果目录
RICH_RESULTS_DIR = "/mnt/sdc1/skx/Time-Series-Library/results_SAFE-Rich/"
# RICH 对应的输出文件名
RICH_OUTPUT_CSV = "SAFE_Rich_results.csv"

# 3. 目标模型列表 (用于从文件夹名中提取到底用了哪个底层模型)
TARGET_MODELS = ['PatchTST', 'DLinear', 'Informer', 'iTransformer', 'TimesNet', 'Transformer']
# ===========================================

def parse_metadata(folder_name):
    """
    解析文件夹名称，提取 Device, Seq_Len 和 具体使用的 Model
    格式示例: long_term_forecast_NE40E_设备83_24_rich_SAFE_...
    """
    # 1. 提取序列长度和设备名
    # 逻辑：寻找 long_term_forecast_ 开头，中间是设备名，紧接着是 _数字_
    pattern = r"^long_term_forecast_(?P<device>.+?)_(?P<seq>\d+)_"
    match = re.search(pattern, folder_name)
    
    device = "Unknown"
    seq_len = 0
    model = "Unknown"
    
    if match:
        device = match.group('device')
        seq_len = int(match.group('seq'))
    
    # 2. 提取底层模型名称 (模糊匹配)
    # 因为文件夹里可能包含 'rich_SAFE' 字样，我们需要知道底层是 PatchTST 还是 Informer
    folder_lower = folder_name.lower()
    for m in TARGET_MODELS:
        if m.lower() in folder_lower:
            model = m
            break
            
    return device, seq_len, model

def process_directory(source_dir, output_csv_name):
    """
    核心处理函数：遍历指定目录并保存为 CSV
    """
    print(f"正在扫描目录: {source_dir} ...")
    
    if not os.path.exists(source_dir):
        print(f"[Error] 目录不存在: {source_dir}")
        return

    sub_folders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
    records = []
    
    for folder_path in sub_folders:
        folder_name = os.path.basename(folder_path)
        
        # 解析元数据
        device, seq_len, model = parse_metadata(folder_name)
        
        # 读取 metrics.npy
        metrics_path = os.path.join(folder_path, "metrics.npy")
        
        if os.path.exists(metrics_path):
            try:
                # TSL 标准: [mae, mse, rmse, mape, mspe]
                metrics = np.load(metrics_path)
                mae = metrics[0]
                mse = metrics[1]
                
                records.append({
                    'Device': device,
                    'Model': model, # 这里记录的是底层模型(如Informer)，表明SAFE路由到了谁
                    'Seq_Len': seq_len,
                    'MSE': mse,
                    'MAE': mae
                })
            except Exception as e:
                print(f"[Warn] 读取失败 {folder_name}: {e}")
    
    # 保存结果
    if records:
        df = pd.DataFrame(records)
        # 排序
        df.sort_values(by=['Device', 'Seq_Len'], inplace=True)
        # 调整列顺序
        df = df[['Device', 'Model', 'Seq_Len', 'MSE', 'MAE']]
        
        df.to_csv(output_csv_name, index=False)
        print(f"[Success] 已保存 {len(df)} 条记录到: {output_csv_name}")
        print("-" * 50)
    else:
        print(f"[Warn] 在 {source_dir} 中未找到有效结果。")

def main():
    print("="*50)
    print("开始提取 SAFE-Basic 与 SAFE-Rich 对比数据")
    print("="*50)

    # 1. 处理 SAFE-Basic
    process_directory(BASIC_RESULTS_DIR, BASIC_OUTPUT_CSV)

    # 2. 处理 SAFE-Rich
    process_directory(RICH_RESULTS_DIR, RICH_OUTPUT_CSV)

    print("\n任务完成！您现在拥有了两份 CSV 文件，可用于画消融实验对比图。")

if __name__ == "__main__":
    main()