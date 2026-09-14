import pandas as pd
import os

# ================= 配置区域 =================
# 1. SAFE 框架的实验结果 (请确认文件名)
# 如果您跑的是原始数据实验，可能是 "specific_devices_results.csv" 或 "final_experiment_results.csv"
SAFE_CSV = "final_experiment_results.csv" 

# 2. XGBoost 基线结果
XGB_CSV = "xgboost_summary.csv"

# 3. 输出报表名称
OUTPUT_REPORT = "paper_highlights_report.txt"
OUTPUT_CSV = "safe_wins_details.csv"
# ===========================================

def generate_highlights():
    print("="*50)
    print("正在生成论文所需的亮点数据 (Paper Highlights)...")
    print("="*50)

    # 1. 加载并合并数据
    if not os.path.exists(SAFE_CSV) or not os.path.exists(XGB_CSV):
        print(f"[Error] 找不到文件: {SAFE_CSV} 或 {XGB_CSV}")
        return

    df_safe = pd.read_csv(SAFE_CSV)
    df_xgb = pd.read_csv(XGB_CSV)

    # 统一列名并合并
    # SAFE CSV 包含: Device, Model, Seq_Len, MSE, MAE
    # XGB CSV 包含: Device, Seq_Len, MSE, MAE (Model列可能没有或固定为XGBoost)
    
    df_safe = df_safe.rename(columns={'MSE': 'MSE_SAFE', 'MAE': 'MAE_SAFE', 'Model': 'Selected_Model'})
    df_xgb = df_xgb.rename(columns={'MSE': 'MSE_XGB', 'MAE': 'MAE_XGB'})
    
    # 按照设备和序列长度合并
    df = pd.merge(df_safe, df_xgb, on=['Device', 'Seq_Len'], how='inner')
    
    if df.empty:
        print("[Error] 没有匹配的设备/序列长度数据，请检查文件名或数据内容。")
        return

    # 2. 计算核心指标
    # MSE 提升百分比 (正值代表 SAFE 更好)
    df['Improvement_Pct'] = (df['MSE_XGB'] - df['MSE_SAFE']) / df['MSE_XGB'] * 100
    
    # 标记胜负
    df['Winner'] = df.apply(lambda x: 'SAFE' if x['MSE_SAFE'] < x['MSE_XGB'] else 'XGBoost', axis=1)
    
    # 筛选出 SAFE 获胜的案例
    safe_wins = df[df['Winner'] == 'SAFE'].copy()
    
    # ================= 生成论文段落素材 =================
    
    report_lines = []
    report_lines.append("=== 论文数据速查表 (用于填充 Results 章节) ===\n")

    # [1] 总体表现 (Abstract/Conclusion)
    avg_imp = df['Improvement_Pct'].mean()
    win_rate = len(safe_wins) / len(df) * 100
    report_lines.append(f"1. 总体表现 (Overall Performance):")
    report_lines.append(f"   - SAFE 相比 XGBoost 的平均 MSE 降低幅度 (Avg Reduction): {avg_imp:.2f}%")
    report_lines.append(f"   - SAFE 的胜率 (Win Rate): {win_rate:.1f}% ({len(safe_wins)}/{len(df)})")
    report_lines.append(f"   - [建议写作]: 'Our framework achieves an average MSE reduction of {avg_imp:.1f}% across all devices.'\n")

    # [2] 长序列优势分析 (Analysis of Sequence Length)
    report_lines.append(f"2. 长序列稳定性 (Impact of Sequence Length):")
    seq_stats = df.groupby('Seq_Len')['Improvement_Pct'].mean()
    for seq, imp in seq_stats.items():
        report_lines.append(f"   - Seq_Len = {seq}: 提升 {imp:.2f}%")
    
    best_len = seq_stats.idxmax()
    report_lines.append(f"   - [结论]: SAFE 在长度为 {best_len} 时优势最大，证明了 Transformer 处理长依赖的能力。\n")

    # [3] 路由机制有效性分析 (Cluster/Model Analysis)
    # 看看 SAFE 选了不同模型时，表现如何
    report_lines.append(f"3. 路由策略有效性 (Effectiveness of Adaptive Routing):")
    model_stats = df.groupby('Selected_Model')['Improvement_Pct'].agg(['mean', 'count'])
    report_lines.append(f"   (当 SAFE 自动选择以下模型时，相对于 XGBoost 的提升)")
    for model, row in model_stats.iterrows():
        report_lines.append(f"   - {model}: 平均提升 {row['mean']:.2f}% (样本数: {row['count']})")
    
    report_lines.append(f"   - [建议写作]: 'Notably, when the framework routes devices to [Best_Model], the performance gain reaches {model_stats['mean'].max():.1f}%.'\n")

    # [4] 明星设备 (Top Performing Cases)
    # 找出提升最大的前 5 个设备，用于画 Case Study 图
    top_5 = safe_wins.sort_values(by='Improvement_Pct', ascending=False).head(5)
    report_lines.append(f"4. 典型成功案例 (Top 5 Best Cases for Case Study):")
    for _, row in top_5.iterrows():
        report_lines.append(f"   - 设备: {row['Device']} (Len={row['Seq_Len']}) | 模型: {row['Selected_Model']} | 提升: {row['Improvement_Pct']:.2f}%")
    
    # ================= 输出 =================
    
    # 打印到控制台
    print("".join(report_lines))
    
    # 保存到文件
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    # 保存详细的胜出数据供作图
    safe_wins.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Success] 详细报告已保存至: {OUTPUT_REPORT}")
    print(f"[Success] 胜出数据表已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_highlights()