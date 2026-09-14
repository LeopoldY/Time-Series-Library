import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features_from_file(file_path, device_name, category):
    """
    提取时间序列特征 (Sparsity, Mean, CV)
    """
    try:
        # 兼容读取
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='gbk')

        target_col = 'w_level4' 
        if target_col not in df.columns:
            return None
        
        data = df[target_col].values
        n_total = len(data)
        
        if n_total < 50: # 过滤极小样本
            return None
        
        n_zeros = np.sum(data == 0)
        sparsity = n_zeros / n_total
        mean_val = np.mean(data)
        std_val = np.std(data)
        cv = std_val / mean_val if mean_val > 1e-6 else 0
        
        return {
            'Device_Name': device_name,
            'Category': category,
            'File_Path': file_path,
            'Root_Path': os.path.dirname(file_path),
            'Data_Path': os.path.basename(file_path),
            'Sparsity': sparsity,
            'Mean_Intensity': mean_val,
            'CV': cv
        }
    except Exception as e:
        print(f"[Error] Extracting {device_name}: {e}")
        return None

def perform_clustering(df_features, n_clusters=3, plot_dir=None):
    """
    K-Means 聚类并打标签，可选导出聚类可视化。
    """
    if df_features.empty:
        return df_features

    scaler = StandardScaler()
    X = df_features[['Sparsity', 'Mean_Intensity', 'CV']]
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_features['Cluster'] = clusters
    
    # 根据稀疏度均值排序打标
    # 稀疏度低(Dense) -> Informer
    # 稀疏度高(Sparse) -> PatchTST
    # 中间(Intermittent) -> iTransformer
    cluster_stats = df_features.groupby('Cluster')['Sparsity'].mean().sort_values()
    
    labels_order = ['Dense', 'Intermittent', 'Sparse']
    mapping = {}
    
    for i, cluster_idx in enumerate(cluster_stats.index):
        if i < len(labels_order):
            mapping[cluster_idx] = labels_order[i]
        else:
            mapping[cluster_idx] = f"Type_{i}"
            
    df_features['Label'] = df_features['Cluster'].map(mapping)

    if plot_dir:
        try:
            save_cluster_plot(df_features, scaler, kmeans, plot_dir)
        except Exception as err:
            print(f"[Warn] Failed to save cluster plot: {err}")

    return df_features

def save_cluster_plot(df_features, scaler, kmeans, plot_dir):
    """
    导出聚类结果 3D 散点图 (Sparsity, Mean_Intensity, CV)。
    """
    if not plot_dir:
        return

    os.makedirs(plot_dir, exist_ok=True)

    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=['Sparsity', 'Mean_Intensity', 'CV'])

    unique_labels = sorted(df_features['Label'].dropna().unique())
    label_count = len(unique_labels)
    palette = plt.cm.get_cmap('Set2', label_count if label_count > 0 else 3)
    color_map = {label: palette(idx) for idx, label in enumerate(unique_labels)}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    legend_handles = []
    legend_labels = []

    for label, group in df_features.groupby('Label'):
        color = color_map.get(label)
        handle = ax.scatter(
            group['Sparsity'],
            group['Mean_Intensity'],
            group['CV'],
            label=label,
            alpha=0.72,
            s=38,
            color=color
        )
        legend_handles.append(handle)
        legend_labels.append(label)

    center_handle = None
    if centers_df is not None:
        center_handle = ax.scatter(
            centers_df['Sparsity'],
            centers_df['Mean_Intensity'],
            centers_df['CV'],
            label='Center',
            color='black',
            marker='X',
            s=90,
            linewidth=1.2
        )
        legend_handles.append(center_handle)
        legend_labels.append('Center')

    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Mean_Intensity')
    ax.set_zlabel('CV')
    ax.set_title('SAFE clustering (3D)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.view_init(elev=25, azim=35)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=8, loc='best')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plot_dir, f"cluster_scatter_{timestamp}.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Cluster Plot] Saved to {plot_path}")