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

# 绘图风格设置
sns.set_theme(style="whitegrid")
# 再次强制设置字体，防止被 seaborn 主题覆盖
plt.rcParams['font.family'] = font_prop.get_name() if os.path.exists(FONT_PATH) else 'serif'

# 定义列名
COL_SEQ_LEN = 'Seq_Len'
COL_METRIC = 'MSE'
COL_DEVICE = 'Device'
COL_METHOD = 'Model'

MODELS_OF_INTEREST = ['Informer', 'PatchTST', 'iTransformer']

# ==========================================
# 1. 读取数据
# ==========================================
try:
    df_safe_all = pd.read_csv('SAFE_experiment_results_all_device.csv')
    df_stacking = pd.read_csv('stacking_results_4_devices.csv')
    df_transformer = pd.read_csv('all_five_models_4_devices.csv')
except FileNotFoundError as e:
    print(f"错误：找不到文件 - {e}")
    exit()

# ==========================================
# 2. 锁定四台目标设备
# ==========================================
if COL_DEVICE in df_stacking.columns:
    target_devices = df_stacking[COL_DEVICE].unique()
    print(f"目标设备: {target_devices}")
else:
    print("错误：Stacking 文件中缺少设备列。")
    exit()

# ==========================================
# 3. 数据预处理 (通用)
# ==========================================

# --- A. SAFE (取 Best) ---
df_safe_filtered = df_safe_all[df_safe_all[COL_DEVICE].isin(target_devices)].copy()
# 选出每个设备每个长度下的最小值
df_safe_best = df_safe_filtered.groupby([COL_SEQ_LEN, COL_DEVICE])[COL_METRIC].min().reset_index()
df_safe_best[COL_METHOD] = 'SAFE'

# --- B. Transformer (保留独立模型) ---
if COL_METHOD in df_transformer.columns:
    df_trans_filtered = df_transformer[
        df_transformer[COL_DEVICE].isin(target_devices) &
        df_transformer[COL_METHOD].isin(MODELS_OF_INTEREST)
    ].copy()
else:
    df_trans_filtered = df_transformer[df_transformer[COL_DEVICE].isin(target_devices)].copy()

# --- C. Stacking ---
df_stacking[COL_METHOD] = 'Stacking'
df_stacking_plot = df_stacking[[COL_SEQ_LEN, COL_METRIC, COL_METHOD, COL_DEVICE]].copy()

# 合并所有数据
df_final = pd.concat([
    df_safe_best,
    df_stacking_plot,
    df_trans_filtered[[COL_SEQ_LEN, COL_METRIC, COL_METHOD, COL_DEVICE]]
], ignore_index=True)

df_final.sort_values(by=COL_SEQ_LEN, inplace=True)

# ==========================================
# 4. 循环为每台设备绘图
# ==========================================
save_dir = 'visualization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for device in target_devices:
    print(f"正在绘制设备: {device} ...")
    
    # 筛选当前设备的数据
    df_device = df_final[df_final[COL_DEVICE] == device].copy()
    
    plt.figure(figsize=(10, 6))
    
    # 绘制箱型图
    ax = sns.boxplot(
        data=df_device,
        x=COL_SEQ_LEN,
        y=COL_METRIC,
        hue=COL_METHOD,
        palette="Set2",
        linewidth=1.2,
        showfliers=False, # 不显示离群点
        width=0.7
    )
    
    # 设置标题和标签 (使用 fontproperties 确保万无一失，或者依赖全局设置)
    # plt.title(f'Performance Comparison on {device}', fontsize=16) 
    # 论文图表通常不需要图内标题，或者标题在图下方，这里留空或注释掉
    
    plt.xlabel('Lookback Length', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 图例设置
    plt.legend(title='Model', title_fontsize=12, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    # 保存文件
    # 处理设备名中可能存在的特殊字符 (如空格)
    safe_device_name = str(device).replace(' ', '_')
    plt.savefig(os.path.join(save_dir, f'boxplot_{safe_device_name}.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, f'boxplot_{safe_device_name}.pdf'), format='pdf')
    
    plt.close() # 关闭当前画布，释放内存

print(f"\n绘图完成！所有图片已保存至 {save_dir}/ 文件夹。")