import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os

# ================= 配置 =================
# 包含所有模型跑分的结果文件 (由 run_all_ada.py 生成)
BENCHMARK_RESULTS_CSV = "final_experiment_results.csv"

# 设备的特征文件 (由 AFD.py 生成的 device_clusters_result.csv)
# 或者你需要重新提取特征，这里假设你已经有这个文件
FEATURE_CSV = "device_clusters_result.csv" 

# 输出结果
OUTPUT_FFORMS_CSV = "fforms_baseline_results.csv"
# =======================================

def main():
    print("="*50)
    print("Running FFORMS (Feature-based Forecast Model Selection)")
    print("="*50)

    # 1. 加载数据
    if not os.path.exists(BENCHMARK_RESULTS_CSV):
        print(f"[Error] Missing {BENCHMARK_RESULTS_CSV}. Run benchmark experiments first!")
        return
    
    df_res = pd.read_csv(BENCHMARK_RESULTS_CSV)
    
    if not os.path.exists(FEATURE_CSV):
        print(f"[Error] Missing {FEATURE_CSV}. Run AFD.py to generate features first!")
        return
        
    df_feat = pd.read_csv(FEATURE_CSV)
    # df_feat 应该包含: Device_Name, Sparsity, Mean_Intensity, CV
    
    # 2. 构建训练样本 (Features, Label)
    # 这里的任务是: 给定设备特征 -> 预测哪个模型最好
    # 我们针对每个 seq_len 分别训练，或者混合训练。这里选择混合训练。
    
    dataset = []
    
    # 获取所有唯一的设备和序列长度组合
    groups = df_res.groupby(['Device', 'Seq_Len'])
    
    for (device, seq_len), group in groups:
        # 找到该组中 MSE 最小的模型
        best_row = group.loc[group['MSE'].idxmin()]
        best_model = best_row['Model']
        min_mse = best_row['MSE']
        
        # 找到该设备的特征
        feat_row = df_feat[df_feat['Device_Name'] == device]
        if feat_row.empty: continue
        
        # 构建样本
        sample = {
            'Device': device,
            'Seq_Len': seq_len,
            'Sparsity': feat_row['Sparsity'].values[0],
            'CV': feat_row['CV'].values[0],
            'Mean': feat_row['Mean_Intensity'].values[0],
            'Best_Model_Label': best_model, # Y (Label)
            'True_Min_MSE': min_mse         # 理想情况下的 loss
        }
        
        # 把该设备所有候选模型的 MSE 也存下来，方便后面评估
        for _, r in group.iterrows():
            sample[f"MSE_{r['Model']}"] = r['MSE']
            
        dataset.append(sample)
        
    df_data = pd.DataFrame(dataset)
    print(f"Constructed Meta-Dataset: {len(df_data)} samples.")
    
    # 3. 训练与评估 (使用 K-Fold 交叉验证模拟泛化能力)
    # FFORMS 使用随机森林作为元学习器
    X = df_data[['Sparsity', 'CV', 'Mean', 'Seq_Len']].values
    y = df_data['Best_Model_Label'].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fforms_mse_list = []
    accuracies = []
    
    print("\nTraining FFORMS Classifier (Random Forest)...")
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # 预测
        pred_models = clf.predict(X_test)
        
        # 计算准确率 (选对模型了吗?)
        acc = accuracy_score(y_test, pred_models)
        accuracies.append(acc)
        
        # 计算 MSE (选出的模型实际表现如何?)
        for i, idx in enumerate(test_idx):
            selected_model = pred_models[i]
            # 从原始数据中查找该模型在该样本上的 MSE
            # 列名格式: MSE_Informer
            col_name = f"MSE_{selected_model}"
            
            # 如果该模型的数据存在
            if col_name in df_data.columns:
                actual_mse = df_data.iloc[idx][col_name]
                # 处理缺失值 (万一该模型没跑成功)
                if pd.isna(actual_mse):
                    # 惩罚项: 如果选了个没跑出结果的模型，取该样本的最大MSE
                    actual_mse = df_data.iloc[idx]['True_Min_MSE'] * 2 
            else:
                actual_mse = 1.0 # 异常
                
            fforms_mse_list.append(actual_mse)

    # 4. 汇总结果
    avg_fforms_mse = np.mean(fforms_mse_list)
    avg_acc = np.mean(accuracies)
    
    print("-" * 50)
    print(f"FFORMS Baseline Results:")
    print(f"   - Model Selection Accuracy: {avg_acc*100:.2f}%")
    print(f"   - Average MSE (FFORMS): {avg_fforms_mse:.6f}")
    print("-" * 50)
    
    # 保存结果供对比
    with open("fforms_summary.txt", "w") as f:
        f.write(f"FFORMS MSE: {avg_fforms_mse}\n")
        f.write(f"FFORMS Accuracy: {avg_acc}\n")

    print(f"Done. Compare this 'Average MSE' with your SAFE result.")

if __name__ == "__main__":
    main()