import os
import glob
import numpy as np
import pandas as pd

# ================= 配置区域 =================
# 结果根目录
ROOT_PATH = "results_original"

# 目标设备列表
DEVICES = ["设备83", "设备69", "设备58", "设备27"]

# 序列长度列表
SEQ_LENS = [3, 6, 12, 24]

# 参与平均的 Transformer 模型
MODELS = ["Informer", "PatchTST", "iTransformer"]
# ===========================================

def get_folder_path(root, device, seq_len, model):
    """
    根据设备、序列长度和模型名，模糊匹配查找文件夹路径
    例如匹配: *NE40E-X3_设备83_3_1_Informer_custom*
    """
    # 构建搜索模式
    # 注意：根据您提供的路径结构，文件名中包含了 device, seq_len, model 等关键词
    # 模式结构：根目录/任意前缀_设备名_序列长度_预测长度(1)_模型名_custom*
    pattern = os.path.join(root, f"*{device}_{seq_len}*{model}_custom*")
    
    # 查找匹配的文件夹
    found_folders = glob.glob(pattern)
    
    if not found_folders:
        return None
    
    # 如果找到多个，默认取第一个（通常应该只有一个）
    return found_folders[0]

def calculate_metrics():
    results_list = []

    print(f"{'Device':<10} | {'SeqLen':<6} | {'Status':<20} | {'MSE':<10} | {'MAE':<10}")
    print("-" * 70)

    for device in DEVICES:
        for seq_len in SEQ_LENS:
            
            preds_collection = [] # 存放每个模型的预测结果
            ground_truth = None   # 存放真实值
            
            missing_models = []

            for model in MODELS:
                folder = get_folder_path(ROOT_PATH, device, seq_len, model)
                
                if folder and os.path.exists(folder):
                    try:
                        # 加载 .npy 文件
                        pred = np.load(os.path.join(folder, 'pred.npy'))
                        true = np.load(os.path.join(folder, 'true.npy'))
                        
                        # 展平数组 (通常结果是 [Batch, Pred_Len, Channel]，展平方便计算)
                        pred = pred.flatten()
                        true = true.flatten()
                        
                        preds_collection.append(pred)
                        
                        # 只需要读取一次真实值 (理论上所有模型的真实值应该是一样的)
                        if ground_truth is None:
                            ground_truth = true
                        
                        # 简单的对齐检查
                        if len(pred) != len(ground_truth):
                            print(f"[Warning] {model} length mismatch: {len(pred)} vs {len(ground_truth)}")

                    except Exception as e:
                        print(f"[Error] loading {folder}: {e}")
                        missing_models.append(model)
                else:
                    missing_models.append(model)

            # --- 计算平均和指标 ---
            if len(preds_collection) == len(MODELS):
                # 1. 将所有模型的预测值堆叠并求平均
                # stack shape: (3, N) -> mean axis 0 -> (N,)
                ensemble_pred = np.mean(np.vstack(preds_collection), axis=0)
                
                # 2. 计算指标
                mse = np.mean((ensemble_pred - ground_truth) ** 2)
                mae = np.mean(np.abs(ensemble_pred - ground_truth))
                
                # 3. 记录结果
                results_list.append({
                    'Device': device,
                    'Seq_Len': seq_len,
                    'Model': 'Naive_Ensemble', # 平均后的名称
                    'MSE': mse,
                    'MAE': mae
                })
                
                print(f"{device:<10} | {seq_len:<6} | {'Success':<20} | {mse:.6f}   | {mae:.6f}")
                
            else:
                print(f"{device:<10} | {seq_len:<6} | Missing: {','.join(missing_models):<11} | N/A        | N/A")

    # --- 保存最终结果 ---
    if results_list:
        df = pd.DataFrame(results_list)
        # 按照设备和序列长度排序
        df = df.sort_values(by=['Device', 'Seq_Len'])
        
        output_file = 'naive_ensemble_results_table.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("\n" + "="*30)
        print(f"计算完成！结果已保存至: {output_file}")
        print("="*30)
        print(df)
    else:
        print("\n未计算出任何结果，请检查路径配置。")

if __name__ == "__main__":
    calculate_metrics()