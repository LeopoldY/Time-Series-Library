import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import glob
import re
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 1. 结果路径
TRANSFORMER_DIR = "/mnt/sdc1/skx/Time-Series-Library/results/"
STACKING_DIR = "/mnt/sdc1/skx/Time-Series-Library/results_stacking/"

# 2. 输出文件
OUTPUT_CSV = "classification_comparison_final.csv"

# 3. 目标设备 (只分析这四台)
TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]

# 4. 目标模型关键词 (用于从文件夹识别)
# 注意：SAFE-Basic 会通过排除 'rich' 且包含 'SAFE' 来识别
BASELINE_MODELS = ['Informer', 'PatchTST', 'iTransformer']
# ===========================================

def calculate_metrics(y_true, y_pred):
    """
    核心指标计算函数
    动态寻找最佳阈值，计算 F1, Recall, Precision, Accuracy
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 真实值二值化 (假设 > min即为故障，处理归一化数据)
    min_val = np.min(y_true)
    # 给一个小 epsilon 防止浮点误差
    binary_true = (y_true > (min_val + 1e-5)).astype(int)
    
    # 如果该样本完全无故障，无法计算 Recall/F1
    if np.sum(binary_true) == 0:
        return None 

    best_f1 = -1
    best_res = {}
    
    # 在预测值的 50% - 99% 分位数之间搜索阈值
    # 对于极度不平衡数据，阈值通常很高
    thresholds = np.percentile(y_pred, np.arange(50, 99.5, 1))
    
    for th in thresholds:
        binary_pred = (y_pred > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(binary_true, binary_pred, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            acc = accuracy_score(binary_true, binary_pred)
            best_res = {
                'Accuracy': acc, 
                'Precision': p, 
                'Recall': r, 
                'F1': f1,
                'Threshold': th
            }
            
    return best_res

def parse_transformer_folder(folder_name):
    """解析 TSLib 结果文件夹名"""
    # 1. 排除 Rich 模型
    if 'rich' in folder_name.lower():
        return None, None
    
    # 2. 识别 SAFE-Basic
    if 'ada' in folder_name:
        return 'SAFE(Ours)', folder_name
    
    # 3. 识别单一基线 (Informer, PatchTST, iTransformer)
    for model in BASELINE_MODELS:
        # 忽略大小写匹配
        if model.lower() in folder_name.lower():
            # 返回标准名称 (例如 iTransformer)
            return model, folder_name
            
    return None, None

def main():
    print("="*50)
    print("Running Classification Comparison (No Rich)")
    print("Targets: SAFE-Basic, Stacking, Informer, PatchTST, iTransformer")
    print("="*50)
    
    results = []

    # ================= 1. 扫描 Transformer & SAFE =================
    print(f"Scanning {TRANSFORMER_DIR} ...")
    trans_folders = glob.glob(os.path.join(TRANSFORMER_DIR, "*"))
    
    for folder in trans_folders:
        if not os.path.isdir(folder): continue
        name = os.path.basename(folder)
        
        # A. 设备过滤
        matched_device = None
        for t in TARGET_DEVICES:
            if t in name:
                matched_device = t
                break
        if not matched_device: continue
        
        # B. 模型识别与过滤
        model_name, _ = parse_transformer_folder(name)
        if not model_name: continue
        
        # C. 提取序列长度 (正则)
        # 格式通常为: ..._序列长度_...
        # 这里尝试匹配 _3_, _6_, _12_, _24_ 等特征
        # TSLib命名: device_seq_label_pred...
        # 假设 seq_len 是设备名后的第一个数字
        try:
            # 这里的正则假设 seq_len 在设备名之后
            seq_match = re.search(r'_(\d+)_', name.replace(matched_device, '')) 
            if not seq_match: continue
            seq_len = int(seq_match.group(1))
        except:
            continue

        # D. 计算指标
        try:
            pred_path = os.path.join(folder, "pred.npy")
            true_path = os.path.join(folder, "true.npy")
            if not os.path.exists(pred_path): continue
            
            pred = np.load(pred_path)
            true = np.load(true_path)
            
            metrics = calculate_metrics(true, pred)
            if metrics:
                metrics.update({
                    'Device': matched_device,
                    'Model': model_name,
                    'Seq_Len': seq_len
                })
                results.append(metrics)
                # print(f"  -> Processed {model_name} on {matched_device} (L={seq_len})")
        except Exception as e:
            # print(f"Error {name}: {e}")
            pass

    # ================= 2. 扫描 Stacking =================
    print(f"Scanning {STACKING_DIR} ...")
    if os.path.exists(STACKING_DIR):
        stack_folders = glob.glob(os.path.join(STACKING_DIR, "*"))
        for folder in stack_folders:
            name = os.path.basename(folder)
            
            # A. 设备过滤
            matched_device = None
            for t in TARGET_DEVICES:
                if t in name:
                    matched_device = t
                    break
            if not matched_device: continue
            
            # B. 提取长度 (格式: Stacking_Device_Seq)
            try:
                seq_len = int(name.split('_')[-1])
            except: continue
            
            # C. 计算指标
            try:
                pred = np.load(os.path.join(folder, "pred.npy"))
                true = np.load(os.path.join(folder, "true.npy"))
                metrics = calculate_metrics(true, pred)
                if metrics:
                    metrics.update({
                        'Device': matched_device,
                        'Model': 'Stacking',
                        'Seq_Len': seq_len
                    })
                    results.append(metrics)
            except: pass
    else:
        print("[Warn] Stacking directory not found.")

    # ================= 3. 汇总与输出 =================
    if results:
        df = pd.DataFrame(results)
        
        # 排序
        df.sort_values(by=['Device', 'Model', 'Seq_Len'], inplace=True)
        
        # 保存明细
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nDetailed metrics saved to: {OUTPUT_CSV}")
        
        # 打印平均性能表 (按模型聚合)
        print("\n" + "="*20 + " CLASSIFICATION PERFORMANCE SUMMARY " + "="*20)
        summary = df.groupby('Model')[['Accuracy', 'Recall', 'Precision', 'F1']].mean().sort_values('F1', ascending=False)
        print(summary)
        
        # 打印提升率 (相对于 Stacking)
        if 'SAFE-Basic' in summary.index and 'Stacking' in summary.index:
            safe_f1 = summary.loc['SAFE-Basic', 'F1']
            stack_f1 = summary.loc['Stacking', 'F1']
            imp = (safe_f1 - stack_f1) / stack_f1 * 100
            print("-" * 60)
            print(f"SAFE-Basic F1 Improvement vs Stacking: {imp:.2f}%")
            print("-" * 60)
            
    else:
        print("[Error] No valid metrics calculated.")

if __name__ == "__main__":
    main()