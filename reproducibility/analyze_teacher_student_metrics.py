#!/usr/bin/env python3
"""
Comprehensive analysis for teacher-student metrics to support paper figures.

Reads all CSVs in a metrics directory (default: teacher_student_metrics/) and produces:
- Time-series overlays per metric comparing TRACER runs (`tracer_*` wandb CSV column prefixes)
- BMA vs EMA at matched update frequencies (when present in run ids)
- TRACER beta sweeps (overlay across β)
- All-runs overlay for each metric with error bands from __MIN/__MAX
- KL-specific summary figures:
  * Early KL slope magnitude (smaller = slower decrease)
  * KL half-life (steps to reach 50% of initial KL)
  * KL AUC across training (smaller = lower on average)
- A summary CSV with per-run, per-metric: last value, AUC, normalized AUC, early slope, half-life

The script mirrors plot styling used in other analysis scripts (Seaborn whitegrid, lineplots with markers).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Heuristic orientation for metrics
LOWER_IS_BETTER_KEYS = {
    "teacher_student_kl_combined",
    "teacher_avg_entropy_combined",
}

ABS_IS_BETTER_KEYS = {
    "teacher_student_accuracy_diff_combined",
}


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    algorithm: Optional[str]  # "tracer" if run id starts with tracer_, else None
    update_type: Optional[str]  # "bma", "ema", or None
    update_freq: Optional[int]
    beta: Optional[float]

    @property
    def paper_label(self) -> str:
        prefix = "TRACER" if self.algorithm == "tracer" else (self.algorithm or self.run_id)
        suffix_parts: List[str] = []
        if self.update_type:
            suffix_parts.append(self.update_type.upper())
        if self.update_freq is not None:
            suffix_parts.append(f"freq={self.update_freq}")
        if self.beta is not None:
            suffix_parts.append(f"β={self.beta}")
        if suffix_parts:
            return f"{prefix}-" + ", ".join(suffix_parts)
        return prefix


RUN_ID_RE = re.compile(r"^(?P<run>[^-]+)\s*-\s*(?P<base>.+)$")


def parse_run_info(run_id: str) -> RunInfo:
    rid = run_id.strip()
    algorithm: Optional[str] = None
    if rid.startswith("tracer"):
        algorithm = "tracer"

    update_type: Optional[str] = None
    if "bma" in rid:
        update_type = "bma"
    elif "ema" in rid:
        update_type = "ema"

    update_freq: Optional[int] = None
    m = re.search(r"up_freq_(\d+)", rid)
    if m:
        try:
            update_freq = int(m.group(1))
        except ValueError:
            update_freq = None

    beta: Optional[float] = None
    b = re.search(r"beta_([0-9]+(?:\.[0-9]+)?)", rid)
    if b:
        try:
            beta = float(b.group(1))
        except ValueError:
            beta = None

    return RunInfo(run_id=rid, algorithm=algorithm, update_type=update_type, update_freq=update_freq, beta=beta)


def is_min_col(col: str) -> bool:
    return col.endswith("__MIN")


def is_max_col(col: str) -> bool:
    return col.endswith("__MAX")


def strip_minmax_suffix(col: str) -> str:
    if col.endswith("__MIN"):
        return col[:-5]
    if col.endswith("__MAX"):
        return col[:-5]
    return col


def load_metric_csv(csv_path: Path) -> Tuple[str, pd.DataFrame]:
    metric_key = csv_path.stem  # e.g., teacher_student_kl_combined
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f"CSV missing 'Step' column: {csv_path}")
    df = df.copy()
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df = df.dropna(subset=["Step"]).sort_values("Step")
    return metric_key, df


def melt_metric(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    # Build base series and optional min/max shading per run
    value_cols = [c for c in df.columns if c != "Step" and not is_min_col(c) and not is_max_col(c)]
    long_rows: List[Dict[str, Any]] = []
    for value_col in value_cols:
        base = value_col
        min_col = f"{base}__MIN"
        max_col = f"{base}__MAX"

        # Extract run id
        m = RUN_ID_RE.match(base)
        if not m:
            # If the pattern is not matched, skip this column
            run_id = base.split(" - ")[0].strip()
        else:
            run_id = m.group("run").strip()

        info = parse_run_info(run_id)

        values = df[["Step", value_col]].rename(columns={value_col: "value"})
        values["metric"] = metric_key
        values["run_id"] = info.run_id
        values["algorithm"] = info.algorithm
        values["update_type"] = info.update_type
        values["update_freq"] = info.update_freq
        values["beta"] = info.beta
        # Optional error bands
        if min_col in df.columns and max_col in df.columns:
            values["min"] = pd.to_numeric(df[min_col], errors="coerce")
            values["max"] = pd.to_numeric(df[max_col], errors="coerce")
        else:
            values["min"] = np.nan
            values["max"] = np.nan

        long_rows.append(values)

    if not long_rows:
        return pd.DataFrame(columns=["Step", "value", "metric", "run_id", "algorithm", "update_type", "update_freq", "beta", "min", "max"])

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    return long_df


def integrate_auc(step: np.ndarray, value: np.ndarray) -> float:
    if len(step) < 2:
        return float("nan")
    return float(np.trapz(value, step))


def compute_early_slope(step: np.ndarray, value: np.ndarray, fraction: float = 0.1, min_points: int = 10) -> float:
    if len(step) < 2:
        return float("nan")
    n = max(min_points, int(len(step) * max(0.0, min(1.0, fraction))))
    n = min(n, len(step))
    x = step[:n]
    y = value[:n]
    if len(x) < 2:
        return float("nan")
    # Linear regression slope
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except Exception:
        slope = float("nan")
    return slope


def compute_half_life(step: np.ndarray, value: np.ndarray) -> float:
    if len(step) == 0:
        return float("nan")
    v0 = value[0]
    if not np.isfinite(v0) or v0 <= 0:
        return float("nan")
    target = 0.5 * v0
    # Find first step where value <= target
    for s, v in zip(step, value):
        if np.isfinite(v) and v <= target:
            return float(s)
    return float("nan")


def summarize_metric(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (metric, run_id), grp in long_df.groupby(["metric", "run_id"], dropna=False):
        g = grp.sort_values("Step")
        step = g["Step"].to_numpy()
        val = pd.to_numeric(g["value"], errors="coerce").to_numpy()
        if len(step) == 0:
            continue
        last_val = float(val[-1]) if len(val) else float("nan")
        auc = integrate_auc(step, val)
        norm_auc = auc / (float(step[-1] - step[0]) + 1e-9)
        slope = compute_early_slope(step, val)
        half = compute_half_life(step, val)

        # Attach metadata (same within group)
        first_row = g.iloc[0]
        rows.append(
            {
                "metric": metric,
                "run_id": run_id,
                "algorithm": first_row.get("algorithm"),
                "update_type": first_row.get("update_type"),
                "update_freq": first_row.get("update_freq"),
                "beta": first_row.get("beta"),
                "last_value": last_val,
                "auc": auc,
                "auc_normalized": norm_auc,
                "early_slope": slope,
                "half_life_step": half,
            }
        )
    return pd.DataFrame(rows)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ema_time_weighted(values: np.ndarray, steps: np.ndarray, momentum: float) -> np.ndarray:
    """Time-weighted EMA where the effective momentum scales with step deltas.

    y[i] = m_eff * y[i-1] + (1 - m_eff) * x[i], where m_eff = momentum ** (step[i] - step[i-1]).
    If x[i] is NaN, carry previous smoothed value forward.
    """
    if len(values) == 0:
        return values
    vals = np.asarray(values, dtype=float)
    stps = np.asarray(steps, dtype=float)
    out = np.empty_like(vals, dtype=float)

    # Seed with first finite value
    finite_idx = np.where(np.isfinite(vals))[0]
    if len(finite_idx) == 0:
        return np.full_like(vals, np.nan)
    first_i = int(finite_idx[0])
    out[:first_i] = vals[first_i]
    out[first_i] = vals[first_i]
    prev_step = stps[first_i]

    for i in range(first_i + 1, len(vals)):
        delta = max(0.0, float(stps[i] - prev_step))
        m_eff = float(momentum ** delta)
        if not np.isfinite(vals[i]):
            out[i] = out[i - 1]
        else:
            out[i] = m_eff * out[i - 1] + (1.0 - m_eff) * vals[i]
        prev_step = stps[i]
    return out


def smooth_long_df(long_df: pd.DataFrame, momentum: float) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    parts: List[pd.DataFrame] = []
    for (metric, run_id), grp in long_df.groupby(["metric", "run_id"], dropna=False):
        g = grp.sort_values("Step").copy()
        steps = g["Step"].to_numpy()
        g["value"] = ema_time_weighted(pd.to_numeric(g["value"], errors="coerce").to_numpy(), steps, momentum)
        if "min" in g.columns:
            g["min"] = ema_time_weighted(pd.to_numeric(g["min"], errors="coerce").to_numpy(), steps, momentum)
        if "max" in g.columns:
            g["max"] = ema_time_weighted(pd.to_numeric(g["max"], errors="coerce").to_numpy(), steps, momentum)
        parts.append(g)
    return pd.concat(parts, axis=0, ignore_index=True)


def plot_time_series_overlay(
    metric_key: str,
    df: pd.DataFrame,
    out_dir: Path,
    title_suffix: str,
    run_filter: Optional[pd.Series] = None,
    color_by: str = "run_id",
    filename_suffix: str = "",
) -> None:
    mdf = df[df["metric"] == metric_key]
    if run_filter is not None:
        mdf = mdf[run_filter.loc[mdf.index] if isinstance(run_filter, pd.Series) else run_filter]
    if mdf.empty:
        return

    # Legend-friendly ordering: sort by algorithm, update_type, freq, beta
    mdf = mdf.sort_values(["algorithm", "update_type", "update_freq", "beta", "run_id", "Step"], na_position="last")

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        # Color by run group
        palette = sns.color_palette("tab10", n_colors=mdf[color_by].nunique())
        for (label, g), color in zip(mdf.groupby(color_by), palette):
            g = g.sort_values("Step")
            ax.plot(g["Step"], g["value"], label=str(label), color=color, marker="o", markersize=2, linewidth=1.5, alpha=0.9)
            # Error band if min/max present
            if g["min"].notna().any() and g["max"].notna().any():
                ax.fill_between(g["Step"], g["min"], g["max"], color=color, alpha=0.12)

        ax.set_title(f"{metric_key.replace('_', ' ').title()} {title_suffix}".strip())
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        # Alphabetically sorted legend by label
        legend_title = color_by.replace("_", " ").title()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            order = np.argsort([str(l) for l in labels])
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            ax.legend(handles, labels, title=legend_title, frameon=False, ncol=2)
        else:
            ax.legend(title=legend_title, frameon=False, ncol=2)
        fig.tight_layout()
        fname = f"{metric_key}{filename_suffix}.png"
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)


def plot_kl_summaries(summary: pd.DataFrame, out_dir: Path) -> None:
    sdf = summary[summary["metric"] == "teacher_student_kl_combined"].copy()
    if sdf.empty:
        return

    # Early slope magnitude (smaller = slower decrease)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sdf_sorted = sdf.sort_values("early_slope", key=lambda s: s.abs())
        labels = [parse_run_info(r).paper_label for r in sdf_sorted["run_id"]]
        ax.barh(labels, sdf_sorted["early_slope"].abs(), color="#1f77b4")
        ax.set_title("KL Early Slope Magnitude (smaller is better – slower decrease)")
        ax.set_xlabel("|Slope|")
        fig.tight_layout()
        fig.savefig(out_dir / "kl_early_slope_magnitude_bar.png", dpi=200)
        plt.close(fig)

    # KL half-life (larger = slower to halve)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sdf_h = sdf.dropna(subset=["half_life_step"]).sort_values("half_life_step", ascending=False)
        labels = [parse_run_info(r).paper_label for r in sdf_h["run_id"]]
        ax.barh(labels, sdf_h["half_life_step"], color="#ff7f0e")
        ax.set_title("KL Half-life (steps to reach 50% of initial KL)")
        ax.set_xlabel("Steps")
        fig.tight_layout()
        fig.savefig(out_dir / "kl_half_life_bar.png", dpi=200)
        plt.close(fig)

    # KL normalized AUC (smaller = better)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sdf_a = sdf.sort_values("auc_normalized")
        labels = [parse_run_info(r).paper_label for r in sdf_a["run_id"]]
        ax.barh(labels, sdf_a["auc_normalized"], color="#2ca02c")
        ax.set_title("KL Normalized AUC across Training (smaller is better)")
        ax.set_xlabel("Normalized AUC")
        fig.tight_layout()
        fig.savefig(out_dir / "kl_auc_normalized_bar.png", dpi=200)
        plt.close(fig)


def generate_plots(long_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    metrics = sorted(long_df["metric"].unique())
    # Global overlays per metric
    for metric_key in metrics:
        plot_time_series_overlay(
            metric_key,
            long_df,
            out_dir,
            title_suffix="– All Runs",
            run_filter=None,
            color_by="run_id",
            filename_suffix="_all_runs",
        )

        # TRACER runs (wandb CSV columns prefixed with tracer_*)
        run_filter = long_df["metric"].eq(metric_key) & long_df["algorithm"].eq("tracer")  # type: ignore
        plot_time_series_overlay(
            metric_key,
            long_df,
            out_dir,
            title_suffix="– TRACER tracked runs",
            run_filter=run_filter,
            color_by="run_id",
            filename_suffix="_tracer_runs",
        )

        # BMA vs EMA matched frequencies
        mdf = long_df[(long_df["metric"] == metric_key)]
        bma_freqs = set(mdf.loc[mdf["update_type"] == "bma", "update_freq"].dropna().astype(int).unique())
        ema_freqs = set(mdf.loc[mdf["update_type"] == "ema", "update_freq"].dropna().astype(int).unique())
        matched = sorted(bma_freqs.intersection(ema_freqs))
        if matched:
            for freq in matched:
                run_filter = (long_df["metric"].eq(metric_key)) & (long_df["update_freq"].astype("float").eq(float(freq))) & (long_df["update_type"].isin(["bma", "ema"]))  # type: ignore
                plot_time_series_overlay(
                    metric_key,
                    long_df,
                    out_dir,
                    title_suffix=f"– BMA vs EMA (freq={freq})",
                    run_filter=run_filter,
                    color_by="run_id",
                    filename_suffix=f"_bma_vs_ema_freq_{freq}",
                )

        # Beta sweeps (β schedule within TRACER)
        has_beta = mdf["beta"].notna().any()
        if has_beta:
            run_filter = (long_df["metric"].eq(metric_key)) & (long_df["beta"].notna())  # type: ignore
            plot_time_series_overlay(
                metric_key,
                long_df,
                out_dir,
                title_suffix="– TRACER β Sweep",
                run_filter=run_filter,
                color_by="run_id",
                filename_suffix="_beta_sweep",
            )


def join_teacher_student_gt_prob(long_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    teach_key = "teacher_gt_prob_img"
    stud_key = "student_gt_prob_img"
    if teach_key not in set(long_df["metric"]) or stud_key not in set(long_df["metric"]):
        return None
    tdf = long_df[long_df["metric"] == teach_key][["Step", "run_id", "value"]].rename(columns={"value": "teacher_gt_prob"})
    sdf = long_df[long_df["metric"] == stud_key][["Step", "run_id", "value"]].rename(columns={"value": "student_gt_prob"})
    j = pd.merge(tdf, sdf, on=["Step", "run_id"], how="inner")
    j["teacher_minus_student"] = j["teacher_gt_prob"] - j["student_gt_prob"]
    return j


def plot_teacher_student_prob_diff(diff_df: pd.DataFrame, out_dir: Path) -> None:
    if diff_df is None or diff_df.empty:
        return
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for run_id, g in diff_df.groupby("run_id"):
            info = parse_run_info(str(run_id))
            label = info.paper_label
            g = g.sort_values("Step")
            ax.plot(g["Step"], g["teacher_minus_student"], label=label, marker="o", markersize=2, linewidth=1.3)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title("Teacher − Student GT Probability (higher suggests stronger teacher guidance)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Teacher − Student GT Prob")
        ax.grid(True, alpha=0.3)
        # Alphabetically sorted legend by label
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            order = np.argsort([str(l) for l in labels])
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            ax.legend(handles, labels, frameon=False, ncol=2)
        else:
            ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "teacher_minus_student_gt_prob_overlay.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive analysis for teacher-student metrics")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="teacher_student_metrics",
        help="Directory containing teacher-student metrics CSVs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="figures_ts",
        help="Directory to save figures and summary CSV",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Seaborn style (e.g., whitegrid, darkgrid, ticks)",
    )
    parser.add_argument(
        "--early_fraction",
        type=float,
        default=0.1,
        help="Fraction of early steps to estimate slope",
    )
    parser.add_argument(
        "--ema_momentum",
        type=float,
        default=0.99,
        help="EMA momentum for smoothing time series (time-weighted as m**Δstep)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    # Load all CSVs
    csv_files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {in_dir}")

    long_frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        metric_key, df = load_metric_csv(csv_path)
        long_df = melt_metric(df, metric_key)
        if not long_df.empty:
            long_frames.append(long_df)

    if not long_frames:
        raise RuntimeError("No usable data parsed from CSVs.")

    all_long = pd.concat(long_frames, axis=0, ignore_index=True)

    # Smooth for cleaner figures
    smoothed_long = smooth_long_df(all_long, momentum=args.ema_momentum)

    # Generate time-series plots from smoothed data
    generate_plots(smoothed_long, out_dir)

    # Compute and save summaries
    summary = summarize_metric(all_long)
    summary.to_csv(out_dir / "teacher_student_metrics_summary.csv", index=False)

    # KL-specific summaries
    plot_kl_summaries(summary, out_dir)

    # Teacher vs Student GT prob difference overlays
    diff_df = join_teacher_student_gt_prob(smoothed_long)
    if diff_df is not None:
        plot_teacher_student_prob_diff(diff_df, out_dir)

    print(f"Saved figures and summaries to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


