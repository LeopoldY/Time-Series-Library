import pandas as pd
import os

# ================= 配置区域 =================
# 输入文件 (由 compare_classification.py 生成)
INPUT_CSV = "classification_comparison.csv"

# 输出文件
OUTPUT_WIN_CSV = "safe_classification_wins.csv"
OUTPUT_REPORT = "safe_classification_report.txt"
# ===========================================

def analyze_wins():
    print("="*50)
    print("正在分析 SAFE 模型的分类性能优势...")
    print("="*50)

    # 1. 加载数据
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] 找不到输入文件: {INPUT_CSV}")
        print("请确保您已经运行了 compare_classification.py 并生成了该文件。")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 检查数据是否为空
    if df.empty:
        print("[Error] CSV 文件为空。")
        return

    # 2. 数据重组 (Pivot)
    # 将长表转换为宽表，以便同一行能对比 SAFE 和 XGBoost
    # 假设 'Model' 列包含 'SAFE (Ours)' 和 'XGBoost'
    # 指标通常包括: Accuracy, Precision, Recall, F1_Score
    try:
        pivot_df = df.pivot_table(
            index=['Device', 'Seq_Len'], 
            columns='Model', 
            values=['F1_Score', 'Accuracy', 'Recall', 'Precision']
        )
    except KeyError as e:
        print(f"[Error] CSV 列名不匹配，缺少必要的列: {e}")
        print(f"当前列名: {df.columns.tolist()}")
        return

    # 扁平化列名 (例如: ('F1_Score', 'XGBoost') -> 'F1_Score_XGBoost')
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # 自动识别 SAFE 和 XGBoost 的列名 (防止由 '(Ours)' 等后缀导致的匹配失败)
    cols = pivot_df.columns
    safe_f1_col = next((c for c in cols if 'F1' in c and 'SAFE' in c), None)
    xgb_f1_col = next((c for c in cols if 'F1' in c and 'XGB' in c), None)
    
    if not safe_f1_col or not xgb_f1_col:
        print("[Error] 无法识别模型列名，请检查 Model 列的内容。")
        print(f"可用列名: {cols}")
        return

    # 3. 计算优势
    # 主要关注 F1-Score 的提升
    pivot_df['F1_Diff'] = pivot_df[safe_f1_col] - pivot_df[xgb_f1_col]
    pivot_df['F1_Improvement_Pct'] = (pivot_df['F1_Diff'] / pivot_df[xgb_f1_col].replace(0, 1e-6)) * 100
    
    # 同时也计算 Recall 的提升 (运维通常也很看重召回率)
    safe_rec_col = next((c for c in cols if 'Recall' in c and 'SAFE' in c), None)
    xgb_rec_col = next((c for c in cols if 'Recall' in c and 'XGB' in c), None)
    if safe_rec_col and xgb_rec_col:
        pivot_df['Recall_Diff'] = pivot_df[safe_rec_col] - pivot_df[xgb_rec_col]

    # 4. 筛选 SAFE 胜出的案例 (F1_Diff > 0)
    wins = pivot_df[pivot_df['F1_Diff'] > 0].copy()
    
    # 按提升幅度排序
    wins.sort_values(by='F1_Improvement_Pct', ascending=False, inplace=True)

    # 5. 生成分析报告
    report_lines = []
    report_lines.append("=== SAFE 模型分类性能优势分析报告 ===\n")
    
    total_cases = len(pivot_df)
    win_count = len(wins)
    win_rate = (win_count / total_cases) * 100 if total_cases > 0 else 0
    
    report_lines.append(f"1. 总体概况:")
    report_lines.append(f"   - 总对比案例数: {total_cases}")
    report_lines.append(f"   - SAFE 胜出案例数 (F1更高): {win_count}")
    report_lines.append(f"   - 胜率: {win_rate:.2f}%\n")
    
    if win_count > 0:
        avg_imp = wins['F1_Improvement_Pct'].mean()
        max_imp = wins['F1_Improvement_Pct'].max()
        report_lines.append(f"2. 提升幅度 (在胜出案例中):")
        report_lines.append(f"   - 平均 F1 提升: +{avg_imp:.2f}%")
        report_lines.append(f"   - 最大 F1 提升: +{max_imp:.2f}%")
        
        # 寻找 Top 5 明星设备
        report_lines.append(f"\n3. Top 5 最佳表现案例 (可用于论文 Case Study):")
        top_5 = wins.head(5)
        for _, row in top_5.iterrows():
            dev = row['Device']
            seq = row['Seq_Len']
            imp = row['F1_Improvement_Pct']
            f1_s = row[safe_f1_col]
            f1_x = row[xgb_f1_col]
            report_lines.append(f"   - 设备: {dev} (Len={seq}) | F1提升: +{imp:.1f}% (SAFE: {f1_s:.3f} vs XGB: {f1_x:.3f})")
            
        # 序列长度分析
        report_lines.append(f"\n4. 各序列长度下的胜出分布:")
        seq_counts = wins['Seq_Len'].value_counts().sort_index()
        for seq, count in seq_counts.items():
            report_lines.append(f"   - Seq_Len {seq}: {count} 个胜出案例")

    else:
        report_lines.append("   [!] 未发现 SAFE 在 F1-Score 上优于 XGBoost 的案例。")
        report_lines.append("   建议检查：1. 是否有些设备 XGBoost 本身已过拟合？ 2. 深度模型是否训练不充分？")

    # 6. 输出与保存
    print("".join(report_lines))
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    # 保存详细胜出表，只保留关键列以便查看
    columns_to_save = ['Device', 'Seq_Len', safe_f1_col, xgb_f1_col, 'F1_Improvement_Pct']
    if safe_rec_col and xgb_rec_col:
        columns_to_save.extend([safe_rec_col, xgb_rec_col, 'Recall_Diff'])
        
    wins_to_save = wins[columns_to_save]
    wins_to_save.to_csv(OUTPUT_WIN_CSV, index=False)
    
    print(f"\n[Success] 详细报告已保存至: {OUTPUT_REPORT}")
    print(f"[Success] 胜出数据表已保存至: {OUTPUT_WIN_CSV}")

if __name__ == "__main__":
    analyze_wins()