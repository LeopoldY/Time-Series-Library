from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DEVICES = ["27", "58", "69", "83"]
DEFAULT_MODELS = ["iTransformer", "PatchTST", "Informer"]
DEFAULT_SEQ_LENS = [3, 6, 12, 24]


def _normalize_device_list(devices: list[str]) -> list[str]:
	"""Accepts either ['27', ...] or ['设备27', ...] and normalizes to '设备XX'."""
	normalized: list[str] = []
	for d in devices:
		d = str(d).strip()
		if not d:
			continue
		normalized.append(d if d.startswith("设备") else f"设备{d}")
	return normalized


def _pivot_models(
	df: pd.DataFrame,
	*,
	device_col: str = "Device",
	seq_col: str = "Seq_Len",
	model_col: str = "Model",
	models: list[str],
	metrics: list[str],
) -> pd.DataFrame:
	sub = df[df[model_col].isin(models)].copy()
	pivot = sub.pivot_table(
		index=[device_col, seq_col],
		columns=model_col,
		values=metrics,
		aggfunc="first",
	)
	# Flatten MultiIndex columns: (metric, model) -> model_metric
	pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
	pivot = pivot.reset_index()
	return pivot


def _ensure_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
	dup_count = int(df.duplicated(keys).sum())
	if dup_count:
		raise ValueError(f"{label} 存在重复键 {keys}：{dup_count} 行")


def build_device_level_table(
	*,
	repo_root: Path,
	devices: list[str],
	models: list[str],
	seq_lens: list[int],
) -> pd.DataFrame:
	devices_norm = _normalize_device_list(devices)

	all_path = repo_root / "all_five_models_4_devices.csv"
	naive_path = repo_root / "naive_ensemble_results_table.csv"
	safe_path = repo_root / "SAFE_experiment_results_all_device.csv"
	stacking_path = repo_root / "stacking_results_4_devices.csv"

	all_df = pd.read_csv(all_path)
	naive_df = pd.read_csv(naive_path)
	safe_df = pd.read_csv(safe_path)
	stacking_df = pd.read_csv(stacking_path)

	# Filter to requested devices/seq
	all_df = all_df[all_df["Device"].isin(devices_norm) & all_df["Seq_Len"].isin(seq_lens)].copy()
	naive_df = naive_df[naive_df["Device"].isin(devices_norm) & naive_df["Seq_Len"].isin(seq_lens)].copy()
	safe_df = safe_df[safe_df["Device"].isin(devices_norm) & safe_df["Seq_Len"].isin(seq_lens)].copy()
	stacking_df = stacking_df[
		stacking_df["Device"].isin(devices_norm) & stacking_df["Seq_Len"].isin(seq_lens)
	].copy()

	metrics = ["MSE", "MAE"]

	# Base transformer models
	base_wide = _pivot_models(all_df, models=models, metrics=metrics)

	# Naive ensemble
	_ensure_unique(naive_df, ["Device", "Seq_Len"], "naive_ensemble_results_table.csv")
	naive_wide = naive_df[["Device", "Seq_Len", "MSE", "MAE"]].copy()
	naive_wide = naive_wide.rename(
		columns={"MSE": "Naive_Ensemble_MSE", "MAE": "Naive_Ensemble_MAE"}
	)

	# Stacking
	_ensure_unique(stacking_df, ["Device", "Seq_Len"], "stacking_results_4_devices.csv")
	stacking_wide = stacking_df[["Device", "Seq_Len", "MSE", "MAE"]].copy()
	stacking_wide = stacking_wide.rename(columns={"MSE": "Stacking_MSE", "MAE": "Stacking_MAE"})

	# SAFE best result: ignore Model field and treat as SAFE_Best
	_ensure_unique(safe_df, ["Device", "Seq_Len"], "SAFE_experiment_results_all_device.csv")
	safe_wide = safe_df[["Device", "Seq_Len", "MSE", "MAE"]].copy()
	safe_wide = safe_wide.rename(columns={"MSE": "SAFE_Best_MSE", "MAE": "SAFE_Best_MAE"})

	# Merge
	out = base_wide.merge(naive_wide, on=["Device", "Seq_Len"], how="left")
	out = out.merge(stacking_wide, on=["Device", "Seq_Len"], how="left")
	out = out.merge(safe_wide, on=["Device", "Seq_Len"], how="left")

	# Order columns
	ordered_cols: list[str] = ["Device", "Seq_Len"]
	for model in models:
		for metric in metrics:
			col = f"{model}_{metric}"
			if col in out.columns:
				ordered_cols.append(col)
	ordered_cols += [
		"Naive_Ensemble_MSE",
		"Naive_Ensemble_MAE",
		"Stacking_MSE",
		"Stacking_MAE",
		"SAFE_Best_MSE",
		"SAFE_Best_MAE",
	]
	out = out[ordered_cols]

	# Sorting: by device number then seq_len
	out = out.copy()
	out["_device_num"] = out["Device"].astype(str).str.replace("设备", "", regex=False).astype(int)
	out = out.sort_values(["_device_num", "Seq_Len"], ascending=[True, True]).drop(columns=["_device_num"])
	out = out.reset_index(drop=True)
	return out


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"从四个CSV抽取设备层面对比数据：iTransformer/PatchTST/Informer + "
			"Naive Ensemble + Stacking + SAFE(忽略Model字段，视为最佳)。"
		)
	)
	parser.add_argument(
		"--repo-root",
		type=Path,
		default=Path(__file__).resolve().parents[1],
		help="Time-Series-Library 根目录（默认：脚本上两级目录）",
	)
	parser.add_argument(
		"--devices",
		nargs="+",
		default=DEFAULT_DEVICES,
		help="设备列表：可传 27 58 或 设备27 设备58（默认：27 58 69 83）",
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=DEFAULT_MODELS,
		help="模型列表（默认：iTransformer PatchTST Informer）",
	)
	parser.add_argument(
		"--seq-lens",
		nargs="+",
		type=int,
		default=DEFAULT_SEQ_LENS,
		help="Seq_Len 列表（默认：3 6 12 24）",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="输出CSV路径（默认输出到 repo-root 下）",
	)
	args = parser.parse_args()

	df = build_device_level_table(
		repo_root=args.repo_root,
		devices=list(args.devices),
		models=list(args.models),
		seq_lens=list(args.seq_lens),
	)

	out_path = args.out
	if out_path is None:
		devices_tag = "_".join(_normalize_device_list(list(args.devices)))
		out_path = args.repo_root / f"device_level_SAFE_best_comparison_{devices_tag}.csv"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)

	print(f"已生成：{out_path}")
	print(f"行数：{len(df)}（设备×Seq_Len）")
	print("列：", ", ".join(df.columns))


if __name__ == "__main__":
	main()
