import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
# 0. 全局设置 & 字体加载
# ==========================================
# 字体文件路径
FONT_PATH = 'visualization/fonts/Times New Roman.ttf'

# 注册本地字体
if os.path.exists(FONT_PATH):
    # 加载字体属性
    font_prop = fm.FontProperties(fname=FONT_PATH)
    # 将字体添加到 matplotlib 的字体管理器中
    fm.fontManager.addfont(FONT_PATH)
    # 设置全局字体为该字体的名称
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"成功加载字体: {font_prop.get_name()} from {FONT_PATH}")
else:
    print(f"警告：未找到字体文件 {FONT_PATH}，将使用默认字体。")
    plt.rcParams['font.family'] = 'serif'

# 文件路径
FILE_SAFE = 'SAFE_experiment_results_all_device.csv'
FILE_STACKING = 'stacking_results_4_devices.csv'
FILE_NAIVE = 'naive_ensemble_results_table.csv'
FILE_TRANSFORMERS = 'all_five_models_4_devices.csv' # 新增：读取独立Transformer结果

# 列名映射
COL_DEVICE = 'Device'
COL_SEQ = 'Seq_Len'
COL_METRIC = 'MSE'
COL_MODEL = 'Model'

# 目标设备
TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]

# 指定要展示的 Transformer 模型 (确保名字与 CSV 中一致)
TRANSFORMER_MODELS = ['Informer', 'PatchTST', 'iTransformer']

# 自定义颜色 (6种颜色区分)
CUSTOM_PALETTE = {
    'SAFE': '#2ca02c',          # Green (重点高亮)
    'Stacking': '#ff7f0e',      # Orange
    'Naive Ensemble': '#1f77b4',# Blue
    'Informer': '#e377c2',      # Pink
    'PatchTST': '#9467bd',      # Purple
    'iTransformer': '#8c564b'   # Brown
}

# 绘图顺序
HUE_ORDER = ['SAFE', 'Stacking', 'Naive Ensemble', 'iTransformer', 'PatchTST', 'Informer']

# ==========================================
# 1. 数据读取与处理
# ==========================================
def load_and_process():
    try:
        df_safe_raw = pd.read_csv(FILE_SAFE)
        df_stack = pd.read_csv(FILE_STACKING)
        df_naive = pd.read_csv(FILE_NAIVE)
        df_trans = pd.read_csv(FILE_TRANSFORMERS)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return None

    # --- A. SAFE (取 Min) ---
    if 'Method' in df_safe_raw.columns: df_safe_raw.rename(columns={'Method': COL_MODEL}, inplace=True)
    df_safe = df_safe_raw[df_safe_raw[COL_DEVICE].isin(TARGET_DEVICES)].copy()
    df_safe = df_safe.groupby([COL_DEVICE, COL_SEQ])[COL_METRIC].min().reset_index()
    df_safe[COL_MODEL] = 'SAFE'

    # --- B. Stacking ---
    if 'Method' in df_stack.columns: df_stack.rename(columns={'Method': COL_MODEL}, inplace=True)
    df_stack[COL_MODEL] = 'Stacking'
    df_stack = df_stack[[COL_DEVICE, COL_SEQ, COL_METRIC, COL_MODEL]]

    # --- C. Naive Ensemble ---
    if 'Method' in df_naive.columns: df_naive.rename(columns={'Method': COL_MODEL}, inplace=True)
    # 确保名字统一
    df_naive[COL_MODEL] = 'Naive Ensemble' 
    df_naive = df_naive[[COL_DEVICE, COL_SEQ, COL_METRIC, COL_MODEL]]

    # --- D. Individual Transformers (新增) ---
    if 'Method' in df_trans.columns: df_trans.rename(columns={'Method': COL_MODEL}, inplace=True)
    # 1. 筛选设备
    df_trans = df_trans[df_trans[COL_DEVICE].isin(TARGET_DEVICES)].copy()
    # 2. 筛选模型 (只保留 Informer, PatchTST, iTransformer)
    df_trans = df_trans[df_trans[COL_MODEL].isin(TRANSFORMER_MODELS)].copy()
    # 3. 如果同一个设置下有多次实验(比如随机种子)，取平均
    df_trans = df_trans.groupby([COL_DEVICE, COL_SEQ, COL_MODEL])[COL_METRIC].mean().reset_index()
    df_trans = df_trans[[COL_DEVICE, COL_SEQ, COL_METRIC, COL_MODEL]]

    # --- E. 合并所有 ---
    df_final = pd.concat([df_safe, df_stack, df_naive, df_trans], ignore_index=True)
    return df_final

# ==========================================
# 2. 绘图逻辑
# ==========================================
def plot_per_device(df):
    if df is None or df.empty: return

    save_dir = 'visualization_bars_6models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    devices = df[COL_DEVICE].unique()

    for device in devices:
        print(f"正在绘制: {device} ...")
        df_sub = df[df[COL_DEVICE] == device].copy()
        df_sub.sort_values(by=COL_SEQ, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid", font="Times New Roman")

        ax = sns.barplot(
            data=df_sub,
            x=COL_SEQ,
            y=COL_METRIC,
            hue=COL_MODEL,
            hue_order=HUE_ORDER,     # 指定柱子的显示顺序
            palette=CUSTOM_PALETTE,  # 指定颜色
            edgecolor='black',
            linewidth=0.8,
            width=0.8                # 调整柱子总宽度
        )

        plt.xlabel('Sequence Length', fontsize=14)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 图例设置 (放在图外或合适位置)
        plt.legend(title='', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        
        safe_name = str(device).replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f'barplot_6models_{safe_name}.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, f'barplot_6models_{safe_name}.pdf'), format='pdf')
        plt.close()

    print(f"\n绘图完成！图片保存在 {save_dir}/ 文件夹。")

if __name__ == "__main__":
    data = load_and_process()
    plot_per_device(data)