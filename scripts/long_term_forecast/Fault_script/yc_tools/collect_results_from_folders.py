import os
import numpy as np
import pandas as pd
import re
import glob

# ================= 配置区域 =================
# 1. 结果文件夹所在的根目录
RESULTS_ROOT = "/mnt/sdc1/skx/Time-Series-Library/results/"

# 2. 输出的 CSV 文件名
OUTPUT_CSV = "all_five_models_fuzzy_match.csv"

# 3. 目标模型列表 (标准名称)
# 脚本会忽略大小写进行搜索，但保存时会使用这里的标准写法
TARGET_MODELS = ['PatchTST', 'DLinear', 'Informer', 'iTransformer', 'TimesNet']
# ===========================================

def match_model_name(folder_name):
    """
    模糊匹配：检查文件夹名中是否包含目标模型名称
    """
    folder_lower = folder_name.lower()
    
    for model in TARGET_MODELS:
        # 将目标模型名转小写进行比对
        if model.lower() in folder_lower:
            return model # 返回标准写法 (例如找到 itransformer 返回 iTransformer)
            
    return "Unknown"

def parse_metadata(folder_name):
    """
    提取 Device 和 Seq_Len，同时调用模糊匹配获取 Model
    """
    # 1. 确定模型
    model_name = match_model_name(folder_name)
    
    # 2. 提取 Device 和 Seq_Len
    # 依然需要正则来区分设备名和序列长度
    # 假设结构仍然是: long_term_forecast_设备名_序列长度_其他参数...
    # 这个正则匹配: "long_term_forecast_" 开头，中间是设备名，然后是 "_数字_" 作为序列长度
    # 这里的关键是利用 "_数字_" 这个特征来截断设备名
    pattern = r"^long_term_forecast_(.+?)_(\d+)_"
    
    match = re.search(pattern, folder_name)
    
    if match:
        device_name = match.group(1)
        seq_len = int(match.group(2))
        return {
            'Device': device_name,
            'Seq_Len': seq_len,
            'Model': model_name
        }
    
    return None

def main():
    print("="*50)
    print("开始搜集结果 (模糊匹配模型名称)...")
    print(f"源目录: {RESULTS_ROOT}")
    print("="*50)

    if not os.path.exists(RESULTS_ROOT):
        print(f"[Error] 找不到目录: {RESULTS_ROOT}")
        return

    sub_folders = [f.path for f in os.scandir(RESULTS_ROOT) if f.is_dir()]
    print(f"扫描到 {len(sub_folders)} 个文件夹。")
    
    records = []
    model_stats = {m: 0 for m in TARGET_MODELS} # 统计每个模型找到多少个
    
    for folder_path in sub_folders:
        folder_name = os.path.basename(folder_path)
        
        # 1. 解析元数据
        metadata = parse_metadata(folder_name)
        
        if not metadata:
            continue
            
        # 如果模型是 Unknown，可能不是我们要找的实验，跳过
        if metadata['Model'] == 'Unknown':
            continue

        # 2. 读取 metrics.npy
        metrics_path = os.path.join(folder_path, "metrics.npy")
        
        if os.path.exists(metrics_path):
            try:
                # TSL 标准: [mae, mse, rmse, mape, mspe]
                metrics = np.load(metrics_path)
                mae = metrics[0]
                mse = metrics[1]
                rmse = metrics[2]
                
                records.append({
                    'Device': metadata['Device'],
                    'Model': metadata['Model'],
                    'Seq_Len': metadata['Seq_Len'],
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                })
                
                # 计数
                model_stats[metadata['Model']] += 1
                
            except Exception as e:
                print(f"[Error] 读取数据失败 {folder_name}: {e}")
        else:
            # 只有文件夹没有 metrics.npy，可能是正在运行或报错中断
            pass

    # 3. 保存与统计
    if records:
        df = pd.DataFrame(records)
        
        # 排序
        df.sort_values(by=['Device', 'Model', 'Seq_Len'], inplace=True)
        
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "-"*50)
        print("【统计报告】")
        for model, count in model_stats.items():
            status = "✅" if count > 0 else "❌"
            print(f"{status} {model:<15}: {count} 条记录")
            
        print("-" * 50)
        print(f"结果已保存至: {os.path.abspath(OUTPUT_CSV)}")
        
    else:
        print("[Warning] 未找到包含目标模型名称的有效实验结果。")

if __name__ == "__main__":
    main()