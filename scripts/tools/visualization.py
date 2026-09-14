import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

def load_and_aggregate(file_path):
    csv_file = file_path
    if not os.path.exists(csv_file):
        raise ValueError("No CSV file found.")


    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    # 仅保留需要列
    cols = ["date", "w_level1", "w_level2", "w_level3", "w_level4"]
    df = df[cols]

    # 转换时间
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df



def aggregate_by_day(df):
    df["date_day"] = df["date"].dt.date  # 提取日期（不带时分秒）

    daily_df = df.groupby("date_day")[["w_level1", "w_level2", "w_level3", "w_level4"]].sum()

    return daily_df


def plot_faults(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(16, 8))

    # 堆叠柱状图
    plt.bar(df.index, df["w_level1"], label="w_level1")
    plt.bar(df.index, df["w_level2"], bottom=df["w_level1"], label="w_level2")
    plt.bar(df.index, df["w_level3"], bottom=df["w_level1"] + df["w_level2"], label="w_level3")
    plt.bar(
        df.index,
        df["w_level4"],
        bottom=df["w_level1"] + df["w_level2"] + df["w_level3"],
        label="w_level4"
    )

    plt.xlabel("Date")
    plt.ylabel("Fault Count")
    plt.title("Daily Aggregated Network Fault Statistics")
    plt.legend()

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Visualization saved to: {output_path}")


def aggregate_all_files(root):
    all_frames = []
    csv_paths = glob.glob(os.path.join(root, "*.csv"))

    if not csv_paths:
        raise ValueError(f"No CSV files found in {root}")

    for file in tqdm(csv_paths, desc="Loading CSVs"):
        raw_df = load_and_aggregate(file)
        all_frames.append(raw_df)

    merged = pd.concat(all_frames, ignore_index=True)
    return aggregate_by_day(merged)


if __name__ == "__main__":
    root = "dataset/fault_selected_cleaned"
    output_dir = os.path.join(root, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    daily_all = aggregate_all_files(root)
    output = os.path.join(output_dir, "all_devices.png")
    plot_faults(daily_all, output)
