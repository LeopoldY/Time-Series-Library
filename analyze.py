import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

class FinalPaperAnalyzer:
    def __init__(self, file_paths, target_devices):
        self.paths = file_paths
        self.target_devices = target_devices
        self.data_dict = {}
        self.merged_df = None
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def _filter_and_label(self, csv_path, model_label_override=None, filter_models=None):
        if not os.path.exists(csv_path): return pd.DataFrame()
        df = pd.read_csv(csv_path)
        
        # 过滤设备
        if 'Device' in df.columns and self.target_devices:
            pattern = '|'.join([re.escape(d) for d in self.target_devices])
            df = df[df['Device'].astype(str).str.contains(pattern, regex=True)].copy()
        
        # 过滤模型
        if filter_models and 'Model' in df.columns:
            lower_targets = [m.lower() for m in filter_models]
            df = df[df['Model'].str.lower().isin(lower_targets)].copy()
            name_map = {m.lower(): m for m in filter_models}
            df['Model'] = df['Model'].str.lower().map(name_map)

        # 重命名
        if model_label_override:
            df['Model'] = model_label_override
        return df

    def load_data(self):
        print("Loading data...")
        # 加载基线
        self.data_dict['transformers'] = self._filter_and_label(
            self.paths['five_models'], filter_models=['Informer', 'PatchTST', 'iTransformer']
        )
        self.data_dict['stacking'] = self._filter_and_label(self.paths['stacking'], "Stacking")
        
        # 加载 SAFE-Basic (我们的主角)
        self.data_dict['safe_basic'] = self._filter_and_label(self.paths['safe_basic'], "SAFE-Basic")
        
        # 注意：不再加载 SAFE-Rich，或者加载但不合并进最终绘图数据
        
        all_dfs = [df for df in self.data_dict.values() if not df.empty]
        if all_dfs:
            self.merged_df = pd.concat(all_dfs, ignore_index=True)
            self.merged_df = self.merged_df[['Device', 'Model', 'Seq_Len', 'MSE', 'MAE']]
            print(f"Loaded {len(self.merged_df)} records.")

    def plot_seq_len_robustness(self, output_file='fig_5_2_seq_len_final.png'):
        """绘制移除 Rich 版本后的最终图表"""
        if self.merged_df is None: return

        plt.figure(figsize=(10, 6))
        
        # 目标模型列表 (只含 Basic)
        target_models = ['SAFE-Basic', 'Stacking', 'Informer', 'PatchTST', 'iTransformer']
        plot_df = self.merged_df[self.merged_df['Model'].isin(target_models)].copy()
        
        # 颜色: SAFE-Basic 用最醒目的红色
        palette = {
            'SAFE-Basic': '#d62728',  # Red
            'Stacking': '#7f7f7f',    # Gray
            'Informer': '#1f77b4',    # Blue
            'PatchTST': '#2ca02c',    # Green
            'iTransformer': '#9467bd' # Purple
        }
        
        sns.lineplot(
            data=plot_df, x='Seq_Len', y='MSE', hue='Model', style='Model',
            markers=True, dashes=False, linewidth=2.5, palette=palette,
            hue_order=['SAFE-Basic', 'PatchTST', 'Informer', 'iTransformer', 'Stacking']
        )
        
        plt.title('Impact of Sequence Length on Prediction Error', fontsize=16)
        plt.xlabel('Input Sequence Length (Hours)', fontsize=14)
        plt.ylabel('MSE (Lower is Better)', fontsize=14)
        plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks([3, 6, 12, 24])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Chart saved to {output_file}")
        plt.close()

if __name__ == "__main__":
    # 配置路径
    BASE_DIR = "." 
    FILES = {
        'five_models': os.path.join(BASE_DIR, "all_five_models_4_devices.csv"),
        'stacking': os.path.join(BASE_DIR, "stacking_results_4_devices.csv"),
        'safe_basic': os.path.join(BASE_DIR, "SAFE_Basic_results.csv"),
        # 'safe_rich': os.path.join(BASE_DIR, "SAFE_Rich_results.csv") # 不再需要
    }
    TARGET_DEVICES = ["设备83", "设备69", "设备58", "设备27"]
    
    analyzer = FinalPaperAnalyzer(FILES, TARGET_DEVICES)
    analyzer.load_data()
    analyzer.plot_seq_len_robustness()