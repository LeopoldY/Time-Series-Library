import os

# ================= 配置区域 =================
# 您的目标文件路径
FILE_PATH = "dataset/UEA_Fault/Fault_TRAIN.ts"
# ===========================================

def count_labels(path):
    if not os.path.exists(path):
        print(f"[Error] 文件不存在: {path}")
        print("请检查路径是否正确，或文件夹名是否拼写错误。")
        return

    print(f"正在读取文件: {path} ...")
    
    pos_count = 0  # 故障样本 (Label = 1)
    neg_count = 0  # 正常样本 (Label = 0)
    total_count = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 跳过空行和 Header (以 @ 开头的行)
            if not line or line.startswith("@"):
                continue
            
            # 2. 解析数据行
            # .ts 文件格式通常为: dim1_data:dim2_data:...:class_label
            # 我们只需要最后一部分
            parts = line.split(':')
            label_str = parts[-1].strip()
            
            # 3. 统计
            # 有时候标签可能是 "1" 或 "1.0"，转为 float 再转 int 比较稳妥
            try:
                label = int(float(label_str))
                if label == 1:
                    pos_count += 1
                elif label == 0:
                    neg_count += 1
                else:
                    print(f"[Warn] 发现未知标签: {label}")
                
                total_count += 1
            except ValueError:
                print(f"[Warn] 无法解析标签: {label_str}")

    # 4. 输出结果
    print("\n" + "="*30)
    print("📊 数据集统计结果")
    print("="*30)
    print(f"总样本数: {total_count}")
    print(f"------------------------------")
    print(f"✅ 正常样本 (Label 0): {neg_count}")
    print(f"❌ 故障样本 (Label 1): {pos_count}")
    print(f"------------------------------")
    
    if total_count > 0:
        neg_ratio = (neg_count / total_count) * 100
        pos_ratio = (pos_count / total_count) * 100
        ratio = neg_count / pos_count if pos_count > 0 else float('inf')
        
        print(f"正常占比: {neg_ratio:.2f}%")
        print(f"故障占比: {pos_ratio:.2f}%")
        print(f"正负比例: 1 : {ratio:.2f} (即每1个故障对应约{ratio:.1f}个正常样本)")
        
        # 建议
        print("\n💡 建议:")
        if ratio > 10:
            print("  您的数据集极度不平衡 (比例 > 1:10)。")
            print("  请务必使用 BCEWithLogitsLoss(pos_weight=tensor([{}]))".format(int(ratio)))
            print("  或者使用 Focal Loss。")
        else:
            print("  数据集平衡性尚可，可以使用标准 Loss 或轻微加权。")
    else:
        print("[Error] 未找到任何数据行，请检查文件格式。")

if __name__ == "__main__":
    count_labels(FILE_PATH)