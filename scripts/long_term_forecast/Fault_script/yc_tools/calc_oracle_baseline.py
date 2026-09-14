import pandas as pd
import os

# ================= 配置 =================
# 包含所有模型跑分的结果文件
# 这是一个包含 4 个设备 * 3 个模型 * 4 个长度 的 CSV
# 如果你没有合并好的，可以手动在这里构造数据，或者把跑出来的结果汇总到一个csv里
INPUT_CSV = "final_experiment_results.csv" 

# 输出
OUTPUT_TXT = "oracle_comparison_report.txt"

# 你的 4 个完备设备名称 (请替换为你实际跑完的设备名部分字符串)
TARGET_DEVICES = [
    "NE40E_设备83",
    "S9300_设备69",
    "S5300_设备58",
    "S9300_设备27"
]
# =======================================

def main():
    print("Computing Oracle Baseline for Limited Devices...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    # 1. 筛选出这 4 个设备的数据
    # 假设 Device 列包含设备名
    mask = df['Device'].apply(lambda x: any(t in x for t in TARGET_DEVICES))
    df_sub = df[mask].copy()
    
    if df_sub.empty:
        print("No data found for target devices.")
        return

    # 2. 找出 SAFE 的结果
    # 假设你的 SAFE 结果里 Model 列叫 'Informer', 'PatchTST' 等，
    # 我们需要知道 SAFE 对这些设备到底选了谁。
    # 这里我们做一个简化：假设 final_experiment_results.csv 里存的已经是 SAFE 选定后跑的结果。
    # 如果该文件里包含了所有 benchmark 结果，我们需要知道 SAFE 的选择。
    
    # 【临时方案】：
    # 假设 INPUT_CSV 里包含了所有模型对这4个设备的跑分。
    # 我们先计算 "理论最优 (Oracle)" 和 "平均集成 (Average)"
    
    report = []
    report.append(f"=== Oracle Baseline Analysis (on {len(TARGET_DEVICES)} devices) ===\n")
    
    # 按设备和序列长度分组
    groups = df_sub.groupby(['Device', 'Seq_Len'])
    
    oracle_mse_list = []
    worst_mse_list = []
    
    details = []
    
    for (dev, seq), group in groups:
        if group.empty: continue
        
        # 找出该组中的最小值 (Oracle) 和最大值 (Worst)
        best_row = group.loc[group['MSE'].idxmin()]
        worst_row = group.loc[group['MSE'].idxmax()]
        
        oracle_mse = best_row['MSE']
        oracle_model = best_row['Model']
        
        oracle_mse_list.append(oracle_mse)
        worst_mse_list.append(worst_row['MSE'])
        
        details.append({
            'Device': dev,
            'Seq': seq,
            'Oracle_Model': oracle_model,
            'Oracle_MSE': oracle_mse,
            'Worst_MSE': worst_row['MSE']
        })

    if not oracle_mse_list:
        print("Not enough data to compute.")
        return

    avg_oracle_mse = sum(oracle_mse_list) / len(oracle_mse_list)
    avg_worst_mse = sum(worst_mse_list) / len(worst_mse_list)
    
    report.append(f"1. 理论极限 (Oracle Baseline):")
    report.append(f"   - 如果每次都选对模型，平均 MSE 可达: {avg_oracle_mse:.6f}")
    report.append(f"   - 如果每次都选错模型，平均 MSE 会是: {avg_worst_mse:.6f}\n")
    
    report.append(f"2. 详细对比 (Check if SAFE matches Oracle):")
    report.append(f"   (请手动对比下表中的 'Oracle_Model' 是否与你 SAFE 框架自动选的模型一致)")
    
    # 打印详细表
    print("".join(report))
    print(f"{'Device':<20} | {'Seq':<4} | {'Oracle Model (Best)':<15} | {'Min MSE':<10}")
    print("-" * 60)
    for d in details:
        print(f"{d['Device']:<20} | {d['Seq']:<4} | {d['Oracle_Model']:<15} | {d['Oracle_MSE']:.6f}")

    # 保存
    pd.DataFrame(details).to_csv("oracle_baseline_details.csv", index=False)
    print("\nDetailed oracle data saved to oracle_baseline_details.csv")

if __name__ == "__main__":
    main()