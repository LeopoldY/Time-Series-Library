import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def clean_zero_fault_intervals(df, days_threshold=14):
    df = df.sort_values("date")
    df["total_fault"] = df[["w_level1", "w_level2", "w_level3", "w_level4"]].sum(axis=1)
    df["is_zero"] = (df["total_fault"] == 0).astype(int)

    # 连续区间标识
    df["segment_id"] = (df["is_zero"].diff() != 0).cumsum()

    segment_info = df.groupby("segment_id")["is_zero"].agg(["sum", "count"])
    zero_segments = segment_info[segment_info["sum"] == segment_info["count"]]

    # 小时数据，14天阈值
    threshold_hours = days_threshold * 24
    remove_segments = zero_segments[zero_segments["count"] >= threshold_hours].index

    removed_ranges = []
    for seg in remove_segments:
        seg_df = df[df["segment_id"] == seg]
        removed_ranges.append((seg_df["date"].min(), seg_df["date"].max()))

    cleaned_df = df[~df["segment_id"].isin(remove_segments)].copy()
    cleaned_df = cleaned_df.drop(columns=["total_fault", "is_zero", "segment_id"])

    return cleaned_df, removed_ranges


def plot_faults(df, save_path):
    plt.figure(figsize=(16, 8))
    plt.bar(df["date"], df["w_level1"], label="w_level1")
    plt.bar(df["date"], df["w_level2"], bottom=df["w_level1"], label="w_level2")
    plt.bar(df["date"], df["w_level3"], bottom=df["w_level1"] + df["w_level2"], label="w_level3")
    plt.bar(
        df["date"],
        df["w_level4"],
        bottom=df["w_level1"] + df["w_level2"] + df["w_level3"],
        label="w_level4"
    )

    plt.xlabel("Time")
    plt.ylabel("Fault Count")
    plt.title("Cleaned Fault Statistics (Daily)")
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    input_root = "dataset/fault_selected"
    output_root = "dataset/fault_selected_cleaned"
    vis_root = "visualization"
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_root, "*.csv"))

    for file in csv_files:
        filename = os.path.basename(file).replace(".csv", "")
        print(f"Processing {filename} ...")

        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        cleaned_df, removed_ranges = clean_zero_fault_intervals(df, days_threshold=14)

        # 保存清洗结果
        cleaned_path = os.path.join(output_root, filename + "_cleaned.csv")
        cleaned_df.to_csv(cleaned_path, index=False)

        # 输出图像
        fig_path = os.path.join(vis_root, filename + "_cleaned.png")
        plot_faults(cleaned_df, fig_path)

        print(f"Cleaned file saved: {cleaned_path}")
        print(f"Visualization saved: {fig_path}")
        if removed_ranges:
            print("Removed zero-fault intervals:")
            for start, end in removed_ranges:
                print(f"  {start} ~ {end}")
        else:
            print("No long zero-fault intervals found.")
