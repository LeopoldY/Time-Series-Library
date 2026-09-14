import os
import re
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = "/mnt/sdc1/skx/Time-Series-Library/dataset/fault_raw/raw_data"
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # safe for headless environments

def find_alarm_files(root):
    p = Path(root)
    files = []
    for ext in ("*.csv", "*.log", "*.txt"):
        files.extend(p.rglob(ext))
    # filter for left/right in filename (case-insensitive)
    return [f for f in files if re.search(r"(left|right)", f.name, re.I)]

def detect_columns(df):
    # timestamp candidates
    ts_rx = re.compile(r"time|date|ts|datetime", re.I)
    alarm_rx = re.compile(r"alarm|type|level|code|event|msg", re.I)

    ts_col = next((c for c in df.columns if ts_rx.search(c)), None)
    alarm_col = next((c for c in df.columns if alarm_rx.search(c)), None)

    # fallback: first column as timestamp, last as message
    if ts_col is None and len(df.columns) >= 1:
        ts_col = df.columns[0]
    if alarm_col is None and len(df.columns) >= 2:
        alarm_col = df.columns[-1]
    return ts_col, alarm_col

def load_file(path):
    try:
        df = pd.read_csv(path, engine="python", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        # try reading as plain text and build a simple dataframe (one line = message)
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        df = pd.DataFrame({"raw": lines})
    if df.empty:
        return None

    ts_col, alarm_col = detect_columns(df)

    # create unified columns
    df = df.copy()
    if ts_col not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
    if "timestamp" not in df or df["timestamp"].isna().all():
        # try to extract timestamp from raw text using a common ISO regex
        iso_rx = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
        if "raw" in df.columns or alarm_col not in df.columns:
            text_col = "raw" if "raw" in df.columns else df.columns[0]
            extracted = df[text_col].astype(str).str.extract(iso_rx)
            if extracted.shape[1] >= 0:
                df["timestamp"] = pd.to_datetime(extracted[0], errors="coerce")

    if alarm_col in df.columns:
        df["alarm_type"] = df[alarm_col].astype(str).str.strip()
    else:
        # fallback: use filename as alarm type
        df["alarm_type"] = Path(path).stem

    df = df[["timestamp", "alarm_type"]].dropna(subset=["timestamp"])
    if df.empty:
        return None
    return df

def combine_and_plot(root_dir):
    files = find_alarm_files(root_dir)
    dfs = []
    for f in files:
        loaded = load_file(f)
        if loaded is not None and not loaded.empty:
            # tag with source side if possible
            side = "left" if re.search(r"left", f.name, re.I) else ("right" if re.search(r"right", f.name, re.I) else "")
            if side:
                loaded["alarm_type"] = loaded["alarm_type"].astype(str) + f" ({side})"
            dfs.append(loaded)

    if not dfs:
        print("No alarm data found under", root_dir)
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    # save combined log
    out_csv = Path(root_dir) / "combined_alarms.csv"
    combined.to_csv(out_csv, index=False)

    # aggregate by hour and alarm type
    combined["hour"] = combined["timestamp"].dt.floor("H")
    grouped = combined.groupby(["hour", "alarm_type"]).size().unstack(fill_value=0).sort_index()

    if grouped.empty:
        print("No aggregated data to plot.")
        return

    # plot stacked bar with one-hour bins
    fig, ax = plt.subplots(figsize=(20, 8))
    grouped.plot(kind="bar", stacked=True, ax=ax, width=0.8, colormap="tab20")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Alarm Count")
    ax.set_title("Stacked Alarm Counts per Hour (left/right combined)")
    ax.legend(title="Alarm Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_png = Path(root_dir) / "combined_alarms_stacked.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved combined CSV to:", out_csv)
    print("Saved stacked plot to:", out_png)

if __name__ == "__main__":
    combine_and_plot(RAW_DATA_DIR)