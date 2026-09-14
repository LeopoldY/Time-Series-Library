from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_long(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric = metric.upper()
    metric_suffix = f"_{metric}"

    value_cols = [c for c in df.columns if c.endswith(metric_suffix)]
    if not value_cols:
        raise ValueError(f"未找到任何以 '{metric_suffix}' 结尾的列。当前列：{list(df.columns)}")

    long_df = df.melt(
        id_vars=["Device", "Seq_Len"],
        value_vars=value_cols,
        var_name="Method",
        value_name=metric,
    )
    long_df["Method"] = long_df["Method"].str.removesuffix(metric_suffix)
    return long_df


def plot_box_by_lookback(
    *,
    input_csv: Path,
    out_dir: Path,
    metrics: list[str],
    font_path: Path,
    font_size: int,
    legend_font_size: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    import seaborn as sns

    df = pd.read_csv(input_csv)
    required = {"Device", "Seq_Len"}
    if not required.issubset(df.columns):
        raise ValueError(f"输入表缺少列 {required}，实际列：{set(df.columns)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # Force using the local Times New Roman font file.
    font_manager.fontManager.addfont(str(font_path))
    font_prop = font_manager.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42

    legend_font_size = legend_font_size or max(8, font_size - 2)

    sns.set_theme(
        style="whitegrid",
        rc={
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        },
    )

    for metric in metrics:
        long_df = _to_long(df, metric)

        # Display names in legend.
        long_df["Method"] = long_df["Method"].replace({"SAFE_Best": "SAFE(Ours)"})
        methods = long_df["Method"].unique()

        plt.figure(figsize=(12, 6.5))
        sns.boxplot(
            data=long_df,
            x="Seq_Len",
            y=metric.upper(),
            hue="Method",
        )

        ax = plt.gca()
        ax.set_xlabel("Lookback Length", fontproperties=font_prop, fontsize=font_size)
        ax.set_ylabel(metric.upper(), fontproperties=font_prop, fontsize=font_size)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
            tick.set_fontsize(font_size)

        # Leave headroom at the top so the legend does not cover data.
        y_col = metric.upper()
        y_max = long_df[y_col].max()
        y_min = long_df[y_col].min()
        span = y_max - y_min
        margin = 0.12 * span if span > 0 else 0.1 * max(1.0, abs(y_max))
        current_top = ax.get_ylim()[1]
        ax.set_ylim(top=max(current_top, y_max + margin))

        leg = ax.legend(
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=len(methods),
            prop=font_prop,
        )
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontproperties(font_prop)
                text.set_fontsize(legend_font_size)

        # No title.
        ax.set_title("")
        plt.tight_layout()

        stem = input_csv.stem
        out_png = out_dir / f"{stem}_box_by_lookback_{metric.upper()}.png"
        out_pdf = out_dir / f"{stem}_box_by_lookback_{metric.upper()}.pdf"
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf)
        plt.close()

        print(f"已输出: {out_png}")
        print(f"已输出: {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw boxplots by lookback (Seq_Len) from a device-level wide table."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("csv_results/device_level_SAFE_best_comparison_设备27_设备58_设备69_设备83.csv"),
        help="设备层对比宽表 CSV 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualization"),
        help="Output directory (default: visualization/)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["MSE", "MAE"],
        help="Metrics to plot (default: MSE MAE)",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("visualization/fonts/Times New Roman.ttf"),
        help="Font file path (default: visualization/fonts/Times New Roman.ttf)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Base font size (default: 22)",
    )
    parser.add_argument(
        "--legend-font-size",
        type=int,
        default=17,
        help="Legend font size (default: font-size - 2)",
    )
    args = parser.parse_args()

    plot_box_by_lookback(
        input_csv=args.input,
        out_dir=args.out_dir,
        metrics=list(args.metrics),
        font_path=args.font_path,
        font_size=args.font_size,
        legend_font_size=args.legend_font_size,
    )


if __name__ == "__main__":
    main()
