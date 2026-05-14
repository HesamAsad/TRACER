#!/usr/bin/env python3
"""
ICLR Paper Analysis: Elegant, grid-free plots with Open Sans and refined colors.

This script produces independent, publication-quality figures:
- original_vs_finetuned_test.pdf
- forgetting_rates.pdf
- spurious_tradeoff_scatter.pdf
- original_train_val_test.pdf
- spurious_test_bars.pdf

It also generates a results summary table.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------
# Global Style & Palette
# ----------------------------
# Refined neutral + accent palette (soft, print-friendly)
COLORS = {
    "ink": "#1F2933",          # Dark grey for text
    "ink_light": "#616E7C",    # For ticks/labels
    "spine": "#8A9199",        # Subtle spines
    "pretrained": "#9AA5B1",   # Neutral grey for baseline bars
    "accent": "#1A73E8",       # Primary accent (blue)
    "accent_alt": "#EF6C00",   # Secondary accent (orange)
    "good": "#2E7D32",         # Green
    "warn": "#D32F2F",         # Red
    "diag": "#D1D5DB"          # Light grey for reference diagonal
}

# Soft categorical palette for methods (max 10 unique, cycling)
PALETTE = [
    "#1A73E8",  # blue
    "#D81B60",  # pink-red
    "#2E7D32",  # green
    "#EF6C00",  # orange
    "#8E24AA",  # purple
    "#00897B",  # teal
    "#5E35B1",  # deep purple
    "#F9A825",  # amber
    "#3949AB",  # indigo
    "#00ACC1",  # cyan
]

# Global rcParams for elegant figures
mpl.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "P052",
        "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"
    ],
    "text.usetex": False,

    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.figsize": (7, 4.5),

    "axes.linewidth": 0.8,
    "axes.edgecolor": COLORS["spine"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.titleweight": "semibold",
    "axes.labelcolor": COLORS["ink"],
    "axes.titlepad": 8,
    "axes.labelpad": 4,

    "xtick.color": COLORS["ink_light"],
    "ytick.color": COLORS["ink_light"],
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,

    "legend.frameon": False,
    "legend.fontsize": 12,

    "grid.alpha": 0.0,  # No grids globally
    "axes.grid": False, # Ensure no grid
    "axes.axisbelow": True,

    "text.color": COLORS["ink"],
})

# ----------------------------
# Data utilities
# ----------------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "Method" not in df.columns:
        raise ValueError("CSV must contain a 'Method' column.")
    if "Image_Test_Original" not in df.columns:
        raise ValueError("CSV must contain 'Image_Test_Original'.")
    if "Image_Test_Colored" not in df.columns:
        raise ValueError("CSV must contain 'Image_Test_Colored'.")
    if "Pre-trained" not in df["Method"].values:
        raise ValueError("CSV must include a 'Pre-trained' row in 'Method'.")
    return df

def calculate_forgetting_rates(df, methods_to_plot=None):
    pretrained_row = df[df["Method"] == "Pre-trained"]
    if pretrained_row.empty:
        raise ValueError("No 'Pre-trained' row found.")
    pretrained_perf = pretrained_row["Image_Test_Original"].iloc[0]

    forgetting_data = []
    for _, row in df.iterrows():
        method = row["Method"]
        if method == "Pre-trained":
            continue
        if methods_to_plot is not None and method not in methods_to_plot:
            continue

        forgetting_rate = pretrained_perf - row["Image_Test_Original"]
        relative_forgetting = (forgetting_rate / pretrained_perf) * 100 if pretrained_perf else np.nan

        forgetting_data.append({
            "Method": method,
            "Forgetting_Rate_Absolute": forgetting_rate,
            "Forgetting_Rate_Relative": relative_forgetting,
            "Original_Performance": row["Image_Test_Original"],
            "Spurious_Performance": row["Image_Test_Colored"],
        })

    # Sort by absolute forgetting descending for nicer bar ordering
    fdf = pd.DataFrame(forgetting_data).sort_values(
        by="Forgetting_Rate_Absolute", ascending=False
    ).reset_index(drop=True)
    return fdf

# ----------------------------
# Aesthetic helpers
# ----------------------------

def _apply_minimalist_axes(ax):
    # No grid, subtle spines, tight ticks
    ax.grid(False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["spine"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8, color=COLORS["ink_light"])
    return ax

def _add_bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.3, color=COLORS["ink_light"], size=8):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=size,
            color=color,
            weight="semibold",
        )

def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Figure creators (independent)
# ----------------------------

def create_original_vs_finetuned_test(df, save_path, methods_to_plot=None):
    # Methods to plot (exclude Pre-trained)
    if methods_to_plot is not None:
        methods = [m for m in df["Method"].tolist() if m in (["Pre-trained"] + methods_to_plot)]
        plot_df = df[df["Method"].isin(methods)].copy()
    else:
        plot_df = df.copy()
    methods = [m for m in plot_df["Method"].tolist() if m != "Pre-trained"]
    finetuned_df = plot_df[plot_df["Method"] != "Pre-trained"]

    pretrained_perf = df.loc[df["Method"] == "Pre-trained", "Image_Test_Original"].values[0]
    finetuned_perfs = finetuned_df["Image_Test_Original"].tolist()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    _apply_minimalist_axes(ax)

    x = np.arange(len(methods))
    width = 0.36

    bars_pre = ax.bar(
        x - width/2, [pretrained_perf]*len(methods), width,
        label="Pre-trained", color=COLORS["pretrained"], edgecolor="none", alpha=0.95
    )
    bars_ft = ax.bar(
        x + width/2, finetuned_perfs, width,
        label="Fine-tuned", color=COLORS["accent"], edgecolor="none", alpha=0.95
    )

    _add_bar_value_labels(ax, bars_pre, fmt="{:.1f}", offset=0.6)
    _add_bar_value_labels(ax, bars_ft, fmt="{:.1f}", offset=0.6)

    # Subtle forgetting connectors
    for i, orig_perf in enumerate(finetuned_perfs):
        forgetting = pretrained_perf - orig_perf
        if np.isfinite(forgetting) and forgetting != 0:
            ax.plot(
                [x[i] - width/2 + width/2, x[i] + width/2 - width/2],
                [pretrained_perf, orig_perf],
                color=COLORS["spine"], linestyle="--", linewidth=0.8, alpha=0.7,
                zorder=1
            )

    ax.set_xlabel("Method", fontsize=12, color=COLORS["ink"])
    ax.set_ylabel("Accuracy (%)", fontsize=12, color=COLORS["ink"])
    ax.set_title("Original Task Performance (Test)", fontsize=14, weight="semibold", color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=10)
    ax.legend(loc="upper right")

    # Y limits with padding
    y_vals = finetuned_perfs + [pretrained_perf]
    y_min = max(0, min(y_vals) - 3)
    y_max = min(100, max(y_vals) + 5)
    ax.set_ylim(y_min, y_max)

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    print(f"Saved: {save_path}")
    return fig

def create_forgetting_rates_barplot(forgetting_df, save_path, methods_to_plot=None):
    fdf = forgetting_df.copy()
    if methods_to_plot is not None:
        fdf = fdf[fdf["Method"].isin(methods_to_plot)].copy()

    # Sort for nicer order
    fdf = fdf.sort_values(by="Forgetting_Rate_Absolute", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_minimalist_axes(ax)

    y_pos = np.arange(len(fdf))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(fdf))]

    bars = ax.barh(
        y_pos, fdf["Forgetting_Rate_Absolute"].values,
        color=colors, edgecolor="none", alpha=0.95, height=0.62
    )

    # Value labels to the right of bars
    for bar, value in zip(bars, fdf["Forgetting_Rate_Absolute"].values):
        x = bar.get_width()
        ax.text(
            x + 0.25, bar.get_y() + bar.get_height()/2,
            f"{value:.1f}%", va="center", ha="left",
            fontsize=10, color=COLORS["ink_light"], weight="semibold"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fdf["Method"].tolist(), fontsize=10)
    ax.set_xlabel("Forgetting Rate (%)", fontsize=12, color=COLORS["ink"])
    ax.set_title("Catastrophic Forgetting (Lower is Better)", fontsize=14, weight="semibold", color=COLORS["ink"])

    # Tight x-limits with padding
    x_max = max(0.1, fdf["Forgetting_Rate_Absolute"].max()) * 1.15
    ax.set_xlim(0, x_max)

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    print(f"Saved: {save_path}")
    return fig

def create_spurious_correlation_analysis(forgetting_df, save_path, methods_to_plot=None):
    fdf = forgetting_df.copy()
    if methods_to_plot is not None:
        fdf = fdf[fdf["Method"].isin(methods_to_plot)].copy()

    fig, ax = plt.subplots(figsize=(10, 7.8))
    _apply_minimalist_axes(ax)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(fdf))]
    xvals = fdf["Original_Performance"].values
    yvals = fdf["Spurious_Performance"].values

    ax.scatter(
        xvals, yvals, s=160,
        c=colors, edgecolors="white", linewidth=1.2, alpha=0.95
    )

    # Labels
    for i, method in enumerate(fdf["Method"].tolist()):
        ax.annotate(
            method, (xvals[i], yvals[i]),
            xytext=(8, 7), textcoords="offset points",
            fontsize=9, color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#E5E7EB", linewidth=0.9, alpha=0.95)
        )

    # Diagonal reference
    min_val = min(np.min(xvals), np.min(yvals)) - 1
    max_val = max(np.max(xvals), np.max(yvals)) + 1
    ax.plot([min_val, max_val], [min_val, max_val], color=COLORS["diag"], linestyle="--", linewidth=1.0)

    ax.set_xlabel("Original Task Accuracy (%)", fontsize=12, color=COLORS["ink"])
    ax.set_ylabel("Spurious Task Accuracy (%)", fontsize=12, color=COLORS["ink"])
    ax.set_title("Performance Trade-off: Original vs. Spurious", fontsize=14, weight="semibold", color=COLORS["ink"])

    # Correlation
    if len(xvals) >= 2:
        corr = np.corrcoef(xvals, yvals)[0, 1]
        ax.text(
            0.02, 0.98, f"r = {corr:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLORS["accent"], linewidth=0.8, alpha=0.15)
        )

    # Axes limits padding
    xpad = max(1.0, (np.max(xvals) - np.min(xvals)) * 0.08) if len(xvals) else 2
    ypad = max(1.0, (np.max(yvals) - np.min(yvals)) * 0.08) if len(yvals) else 2
    ax.set_xlim(np.min(xvals) - xpad, np.max(xvals) + xpad)
    ax.set_ylim(np.min(yvals) - ypad, np.max(yvals) + ypad)

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    print(f"Saved: {save_path}")
    return fig

def create_original_train_val_test(df, save_path, methods_to_plot=None):
    plot_df = df.copy()
    if methods_to_plot is not None:
        plot_df = plot_df[plot_df["Method"].isin(["Pre-trained"] + methods_to_plot)]

    methods = plot_df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.5))
    _apply_minimalist_axes(ax)

    bars_train = ax.bar(
        x - width, plot_df["Image_Train_Original"], width,
        label="Train", color=COLORS["accent"], edgecolor="none", alpha=0.95
    )
    bars_val = ax.bar(
        x, plot_df["Image_Val_Original"], width,
        label="Validation", color=COLORS["good"], edgecolor="none", alpha=0.95
    )
    bars_test = ax.bar(
        x + width, plot_df["Image_Test_Original"], width,
        label="Test", color=COLORS["accent_alt"], edgecolor="none", alpha=0.95
    )

    for bars in (bars_train, bars_val, bars_test):
        _add_bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.6)

    ax.set_title("Original Dataset Performance", fontsize=14, weight="semibold", color=COLORS["ink"])
    ax.set_ylabel("Accuracy (%)", fontsize=12, color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10, ha="right", fontsize=10)
    ax.legend(loc="upper right")

    # Y limits
    y_vals = []
    for c in ["Image_Train_Original", "Image_Val_Original", "Image_Test_Original"]:
        y_vals.extend(plot_df[c].tolist())
    ax.set_ylim(max(0, min(y_vals) - 3), min(100, max(y_vals) + 5))

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    print(f"Saved: {save_path}")
    return fig

def create_spurious_test_bars(df, save_path, methods_to_plot=None):
    plot_df = df.copy()
    if methods_to_plot is not None:
        plot_df = plot_df[plot_df["Method"].isin(["Pre-trained"] + methods_to_plot)]

    methods = plot_df["Method"].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    _apply_minimalist_axes(ax)

    bars = ax.bar(
        methods, plot_df["Image_Test_Colored"],
        color=colors, edgecolor="none", alpha=0.95
    )

    _add_bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.6)

    ax.set_title("Spurious (Colored) Dataset Performance (Test)", fontsize=14, weight="semibold", color=COLORS["ink"])
    ax.set_ylabel("Accuracy (%)", fontsize=12, color=COLORS["ink"])
    ax.set_xlabel("Method", fontsize=12, color=COLORS["ink"])
    ax.tick_params(axis="x", rotation=10)

    # Y limits
    vals = plot_df["Image_Test_Colored"].tolist()
    ax.set_ylim(max(0, min(vals) - 3), min(100, max(vals) + 5))

    _ensure_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    print(f"Saved: {save_path}")
    return fig

# ----------------------------
# Summary table
# ----------------------------

def generate_summary_table(df, forgetting_df, save_path='tables/results_summary.txt', methods_to_plot=None):
    fdf = forgetting_df.copy()
    if methods_to_plot is not None:
        fdf = fdf[fdf["Method"].isin(methods_to_plot)]

    pretrained_perf = df.loc[df["Method"] == "Pre-trained", "Image_Test_Original"].iloc[0]
    colored_pretrained = df.loc[df["Method"] == "Pre-trained", "Image_Test_Colored"].iloc[0]

    summary = []
    summary.append("="*49)
    summary.append("RESULTS SUMMARY FOR ICLR PAPER")
    summary.append("="*49)
    summary.append("")
    summary.append("Table 1: Performance Comparison (%)")
    summary.append("Method              Original(Test)    Colored(Test)    Forgetting(Abs)")
    summary.append("-"*68)
    summary.append(f"Pre-trained         {pretrained_perf:6.1f}           {colored_pretrained:6.1f}            0.0")

    for _, row in fdf.iterrows():
        summary.append(f"{row['Method']:<18} {row['Original_Performance']:6.1f}           {row['Spurious_Performance']:6.1f}           {row['Forgetting_Rate_Absolute']:6.1f}")

    # Optional key findings if methods exist
    findings = []
    if "Direct FT" in fdf["Method"].values:
        drop = fdf.loc[fdf["Method"] == "Direct FT", "Forgetting_Rate_Absolute"].iloc[0]
        findings.append(f"Direct fine-tuning shows catastrophic forgetting ({drop:.1f}% drop).")
    if "Static Distill" in fdf["Method"].values:
        sd = fdf.loc[fdf["Method"] == "Static Distill", "Forgetting_Rate_Absolute"].iloc[0]
        findings.append(f"Static Distillation reduces forgetting to {sd:.1f}%.")
    if "Dynamic Distill" in fdf["Method"].values:
        dd = fdf.loc[fdf["Method"] == "Dynamic Distill", "Forgetting_Rate_Absolute"].iloc[0]
        findings.append(f"Dynamic Distillation reduces forgetting to {dd:.1f}%.")

    if len(findings) > 0:
        summary.append("")
        summary.append("Key Findings:")
        for i, line in enumerate(findings, start=1):
            summary.append(f"{i}. {line}")

    text = "\n".join(summary)
    _ensure_dir(save_path)
    with open(save_path, "w") as f:
        f.write(text)
    print(f"Saved summary: {save_path}")
    return text

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results for ICLR paper (elegant, grid-free plots)")
    parser.add_argument("--csv_path", default="experiment_results.csv", help="Path to CSV file with results")
    parser.add_argument("--output_dir", default=".", help="Output directory for figures and tables")
    parser.add_argument(
        "--methods", nargs="+", type=str,
        default=["Direct FT", "L2 Reg", "Static Distill", "Dynamic Distill"],
        help='List of methods to plot (excluding "Pre-trained"). If not specified, plot all.'
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    # Load data
    print("Loading experimental data...")
    df = load_data(args.csv_path)
    print(f"Loaded data for {len(df)} rows")

    # Determine methods to plot
    if args.methods is not None and len(args.methods) > 0:
        methods_to_plot = args.methods
        print(f"Plotting selected methods: {methods_to_plot}")
    else:
        methods_to_plot = df[df["Method"] != "Pre-trained"]["Method"].tolist()
        print(f"Plotting all methods: {methods_to_plot}")

    # Calculate forgetting rates
    print("Calculating forgetting rates...")
    forgetting_df = calculate_forgetting_rates(df, methods_to_plot=methods_to_plot)
    print("Forgetting rates (abs, rel):")
    if not forgetting_df.empty:
        print(forgetting_df[["Method", "Forgetting_Rate_Absolute", "Forgetting_Rate_Relative"]])

    # Output directories
    figures_dir = Path(args.output_dir) / "toy_experiment_figures_reproducible"
    tables_dir = Path(args.output_dir) / "toy_experiment_tables_reproducible"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Generate independent, publication-quality plots
    print("\nGenerating elegant, grid-free plots...")
    create_original_vs_finetuned_test(
        df, figures_dir / "original_vs_finetuned_test.pdf", methods_to_plot=methods_to_plot
    )
    create_forgetting_rates_barplot(
        forgetting_df, figures_dir / "forgetting_rates.pdf", methods_to_plot=methods_to_plot
    )
    create_spurious_correlation_analysis(
        forgetting_df, figures_dir / "spurious_tradeoff_scatter.pdf", methods_to_plot=methods_to_plot
    )
    create_original_train_val_test(
        df, figures_dir / "original_train_val_test.pdf", methods_to_plot=methods_to_plot
    )
    create_spurious_test_bars(
        df, figures_dir / "spurious_test_bars.pdf", methods_to_plot=methods_to_plot
    )

    # Generate summary table
    generate_summary_table(
        df, forgetting_df, tables_dir / "results_summary.txt", methods_to_plot=methods_to_plot
    )

    print("\nAnalysis complete! Generated files:")
    print(f"- Figures: {figures_dir}")
    print(f"- Tables: {tables_dir}")
    print(f"- Summary: {tables_dir / 'results_summary.txt'}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()