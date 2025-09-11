#!/usr/bin/env python3
"""
Analyze teacher update frequency ablation and generate plots for ID, OOD, and each dataset.

Inputs
- CSV file like bma_teacher_update_frequency_ablation.csv with columns:
  Name,
  ImageNet Accuracy, Avg OOD Acc,
  ImageNetV2 Accuracy, ImageNetR Accuracy, ImageNetA Accuracy, ImageNetSketch Accuracy, ObjectNet Accuracy,
  bma_teacher_up_freq,
  ImageNet ECE, ImageNetV2 ECE, ImageNetR ECE, ImageNetA ECE, ImageNetSketch ECE, ObjectNet ECE, Avg OOD ECE

Outputs (default: figures_bma/)
- id_accuracy_ece_vs_freq.png
- ood_avg_accuracy_ece_vs_freq.png
- dataset_<name>_accuracy_ece_vs_freq.png (for each OOD dataset)
- ood_per_dataset_accuracy_vs_freq.png
- ood_per_dataset_ece_vs_freq.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ID_ACC_COL = "ImageNet Accuracy"
ID_ECE_COL = "ImageNet ECE"
OOD_AVG_ACC_COL = "Avg OOD Acc"
OOD_AVG_ECE_COL = "Avg OOD ECE"
FREQ_COL = "bma_teacher_up_freq"

DATASET_COLUMNS: Dict[str, Tuple[str, str]] = {
    "ImageNetV2": ("ImageNetV2 Accuracy", "ImageNetV2 ECE"),
    "ImageNetR": ("ImageNetR Accuracy", "ImageNetR ECE"),
    "ImageNetA": ("ImageNetA Accuracy", "ImageNetA ECE"),
    "ImageNetSketch": ("ImageNetSketch Accuracy", "ImageNetSketch ECE"),
    "ObjectNet": ("ObjectNet Accuracy", "ObjectNet ECE"),
}


def validate_columns(df: pd.DataFrame) -> None:
    required = {FREQ_COL, ID_ACC_COL, ID_ECE_COL, OOD_AVG_ACC_COL, OOD_AVG_ECE_COL}
    for _, (acc_col, ece_col) in DATASET_COLUMNS.items():
        required.add(acc_col)
        required.add(ece_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df[FREQ_COL] = pd.to_numeric(df[FREQ_COL], errors="coerce")
    df = df.dropna(subset=[FREQ_COL])
    df = df.sort_values(FREQ_COL)
    validate_columns(df)
    return df


def save_id_plots(df: pd.DataFrame, out_dir: Path, style: str = "whitegrid") -> None:
    with sns.axes_style(style):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

        sns.lineplot(data=df, x=FREQ_COL, y=ID_ACC_COL, marker="o", ax=axes[0], color="#1f77b4")
        axes[0].set_title("ID Accuracy vs Teacher Update Frequency")
        axes[0].set_xlabel("Teacher Update Steps")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, alpha=0.3)

        sns.lineplot(data=df, x=FREQ_COL, y=ID_ECE_COL, marker="o", ax=axes[1], color="#d62728")
        axes[1].set_title("ID ECE vs Teacher Update Frequency")
        axes[1].set_xlabel("Teacher Update Steps")
        axes[1].set_ylabel("ECE")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "id_accuracy_ece_vs_freq.png", dpi=200)
        plt.close(fig)


def save_ood_avg_plots(df: pd.DataFrame, out_dir: Path, style: str = "whitegrid") -> None:
    with sns.axes_style(style):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

        sns.lineplot(data=df, x=FREQ_COL, y=OOD_AVG_ACC_COL, marker="o", ax=axes[0], color="#2ca02c")
        axes[0].set_title("Avg OOD Accuracy vs Teacher Update Frequency")
        axes[0].set_xlabel("Teacher Update Steps")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, alpha=0.3)

        sns.lineplot(data=df, x=FREQ_COL, y=OOD_AVG_ECE_COL, marker="o", ax=axes[1], color="#ff7f0e")
        axes[1].set_title("Avg OOD ECE vs Teacher Update Frequency")
        axes[1].set_xlabel("Teacher Update Steps")
        axes[1].set_ylabel("ECE")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "ood_avg_accuracy_ece_vs_freq.png", dpi=200)
        plt.close(fig)


def save_per_dataset_plots(df: pd.DataFrame, out_dir: Path, style: str = "whitegrid") -> None:
    for dataset_name, (acc_col, ece_col) in DATASET_COLUMNS.items():
        with sns.axes_style(style):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

            sns.lineplot(data=df, x=FREQ_COL, y=acc_col, marker="o", ax=axes[0])
            axes[0].set_title(f"{dataset_name} Accuracy vs Teacher Update Frequency")
            axes[0].set_xlabel("Teacher Update Steps")
            axes[0].set_ylabel("Accuracy")
            axes[0].grid(True, alpha=0.3)

            sns.lineplot(data=df, x=FREQ_COL, y=ece_col, marker="o", ax=axes[1], color="#d62728")
            axes[1].set_title(f"{dataset_name} ECE vs Teacher Update Frequency")
            axes[1].set_xlabel("Teacher Update Steps")
            axes[1].set_ylabel("ECE")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            out_path = out_dir / f"dataset_{dataset_name.lower()}_accuracy_ece_vs_freq.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def save_overlay_plots(df: pd.DataFrame, out_dir: Path, style: str = "whitegrid") -> None:
    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        palette = sns.color_palette("tab10", n_colors=len(DATASET_COLUMNS))
        for (dataset_name, (acc_col, _)), color in zip(DATASET_COLUMNS.items(), palette):
            sns.lineplot(data=df, x=FREQ_COL, y=acc_col, marker="o", ax=ax, label=f"{dataset_name}", color=color)
        ax.set_title("OOD Per-Dataset Accuracy vs Teacher Update Frequency")
        ax.set_xlabel("Teacher Update Steps")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Dataset", frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "ood_per_dataset_accuracy_vs_freq.png", dpi=200)
        plt.close(fig)

    with sns.axes_style(style):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        palette = sns.color_palette("tab10", n_colors=len(DATASET_COLUMNS))
        for (dataset_name, (_, ece_col)), color in zip(DATASET_COLUMNS.items(), palette):
            sns.lineplot(data=df, x=FREQ_COL, y=ece_col, marker="o", ax=ax, label=f"{dataset_name}", color=color)
        ax.set_title("OOD Per-Dataset ECE vs Teacher Update Frequency")
        ax.set_xlabel("Teacher Update Steps")
        ax.set_ylabel("ECE")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Dataset", frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "ood_per_dataset_ece_vs_freq.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for teacher update frequency ablation")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="bma_teacher_update_frequency_ablation.csv",
        help="Path to CSV file (default: bma_teacher_update_frequency_ablation.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="figures_bma_up_freq",
        help="Directory to save figures (default: figures_bma)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Seaborn style (e.g., whitegrid, darkgrid, ticks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)

    save_id_plots(df, out_dir, style=args.style)
    save_ood_avg_plots(df, out_dir, style=args.style)
    save_per_dataset_plots(df, out_dir, style=args.style)
    save_overlay_plots(df, out_dir, style=args.style)

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


