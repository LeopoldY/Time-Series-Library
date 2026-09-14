import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
# 0. 全局设置 & 字体加载
# ==========================================
FONT_PATH = 'visualization/fonts/Times New Roman.ttf'

# 注册本地字体
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    print(f"警告：未找到字体文件 {FONT_PATH}，使用默认 Serif 字体。")
    plt.rcParams['font.family'] = 'serif'

# 绘图风格
sns.set_theme(style="whitegrid")
# 再次强制设置字体，防止被主题覆盖
if os.path.exists(FONT_PATH):
    plt.rcParams['font.family'] = font_prop.get_name()

# 列名定义
COL_SEQ_LEN = 'Seq_Len'
COL_METRIC = 'MSE'
COL_DEVICE = 'Device'
COL_METHOD = 'Model'

# 目标设备 & 模型
TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]
TRANSFORMER_MODELS = ['Informer', 'PatchTST', 'iTransformer']

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

# 数据清洗：去除设备名称可能存在的空格
for df in [df_safe_all, df_stacking, df_transformer]:
    if COL_DEVICE in df.columns:
        df[COL_DEVICE] = df[COL_DEVICE].astype(str).str.strip()

# ==========================================
# 2. 数据处理 (严格按设备抓取)
# ==========================================

# --- A. SAFE (取每个设备、每个长度下的 Best) ---
# 1. 严格筛选目标设备
df_safe_filtered = df_safe_all[df_safe_all[COL_DEVICE].isin(TARGET_DEVICES)].copy()
# 2. 核心步骤：按 [设备, 序列长度] 分组，取 Min MSE
# 这样确保了 SAFE 的数据点是严格对应到具体设备的
df_safe_best = df_safe_filtered.groupby([COL_DEVICE, COL_SEQ_LEN])[COL_METRIC].min().reset_index()
df_safe_best[COL_METHOD] = 'SAFE'

# --- B. Transformer (筛选模型 & 设备) ---
if COL_METHOD in df_transformer.columns:
    df_trans_filtered = df_transformer[
        df_transformer[COL_DEVICE].isin(TARGET_DEVICES) &
        df_transformer[COL_METHOD].isin(TRANSFORMER_MODELS)
    ].copy()
else:
    df_trans_filtered = df_transformer[df_transformer[COL_DEVICE].isin(TARGET_DEVICES)].copy()

# --- C. Stacking (筛选设备) ---
df_stacking_filtered = df_stacking[df_stacking[COL_DEVICE].isin(TARGET_DEVICES)].copy()
df_stacking_filtered[COL_METHOD] = 'Stacking'

# 合并所有数据
df_final = pd.concat([
    df_safe_best,
    df_stacking_filtered[[COL_DEVICE, COL_METRIC, COL_METHOD]], # Stacking 可能没有 SeqLen 列，如果有也无所谓，箱型图只看 MSE
    df_trans_filtered[[COL_DEVICE, COL_METRIC, COL_METHOD]]
], ignore_index=True)

# ==========================================
# 3. 绘制箱型图 (X=Device)
# ==========================================
save_dir = 'visualization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.figure(figsize=(12, 7))

# 自定义颜色顺序
hue_order = ['SAFE', 'Stacking', 'iTransformer', 'PatchTST', 'Informer']

# 绘图
ax = sns.boxplot(
    data=df_final,
    x=COL_DEVICE,      # X轴：设备
    y=COL_METRIC,      # Y轴：MSE
    hue=COL_METHOD,    # 颜色：模型
    hue_order=hue_order,
    palette="Set2",
    linewidth=1.2,
    showfliers=False,  # 隐藏离群点，让分布更清晰
    width=0.75
)

# 标签设置
plt.xlabel('Device ID', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 图例设置
plt.legend(title='Model', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

# 保存图片
plt.savefig(os.path.join(save_dir, 'boxplot_by_device.png'), dpi=300)
plt.savefig(os.path.join(save_dir, 'boxplot_by_device.pdf'), format='pdf')

plt.show()

print(f"绘图完成！图片已保存至 {save_dir}/boxplot_by_device.png")