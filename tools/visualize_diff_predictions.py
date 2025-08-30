import os
import csv
import argparse

from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib import gridspec as _gridspec

# Make repo imports available if executed directly
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets_.imagenet_classnames import get_classnames


# Adopt plot styles similar to toy_experiment_repr_analyze.py
_PREFERRED_SANS_SERIF = ['Inter', 'Roboto', 'SF Pro Display', 'Avenir', 'DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']

def _select_preferred_sans_serif():
    for _name in _PREFERRED_SANS_SERIF:
        try:
            _fm.findfont(_name, fallback_to_default=False)
            return _name
        except Exception:
            continue
    return 'DejaVu Sans'

_SELECTED_SANS = _select_preferred_sans_serif()

plt.rcParams.update({
    'font.family': _SELECTED_SANS,
    'font.sans-serif': [_SELECTED_SANS] + _PREFERRED_SANS_SERIF,
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'axes.linewidth': 1.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#222222',
    'axes.axisbelow': True,
    'axes.labelpad': 6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.3,
    'ytick.major.width': 1.3,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#DDDDDD',
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.figsize': (12.0, 3.8),
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    'axes.grid': False,
})

def read_scores(csv_path: str) -> List[dict]:
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data_rows = rows[1:]

    # Identify column indices
    col_idx = {name: i for i, name in enumerate(header)}
    # find prob columns
    zs_cols = [i for i, n in enumerate(header) if n.startswith('zs_prob_')]
    ft_cols = [i for i, n in enumerate(header) if n.startswith('ft_prob_')]

    records = []
    for r in data_rows:
        rec = {
            'global_index': int(r[col_idx['global_index']]),
            'image_path': r[col_idx['image_path']],
            'gt': int(r[col_idx['gt']]),
            'pred_zs': int(r[col_idx['pred_zs']]),
            'pred_ft': int(r[col_idx['pred_ft']]),
            'zs_probs': np.array([float(r[i]) for i in zs_cols], dtype=np.float32),
            'ft_probs': np.array([float(r[i]) for i in ft_cols], dtype=np.float32),
        }
        records.append(rec)
    return records


def topk(probs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-probs, kth=min(k, len(probs)-1))[:k]
    idx = idx[np.argsort(-probs[idx])]
    vals = probs[idx]
    return idx, vals


def plot_sample(fig_out: str,
                image_path: str,
                zs_probs: np.ndarray,
                ft_probs: np.ndarray,
                classnames: List[str],
                gradcam_image_path: str,
                title: str = '') -> None:
    img = Image.open(image_path).convert('RGB')

    zs_idx, zs_vals = topk(zs_probs, k=5)
    ft_idx, ft_vals = topk(ft_probs, k=5)

    # Build figure with 4 columns; make Grad-CAM composite wider
    fig = plt.figure(figsize=(12.0, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.25, 1.25, 2.6])

    # 1) Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Original', fontsize=12, pad=2)

    # 2) ZS top-5 bar (top class at top)
    ax2 = fig.add_subplot(gs[0, 1])
    labels_zs = [classnames[i] if i < len(classnames) else str(i) for i in zs_idx]
    y_pos = np.arange(len(zs_vals))
    ax2.barh(y_pos, zs_vals, color='#1f77b4')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_zs, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('ZS prob')
    ax2.set_title('Zero-shot Top-5', fontsize=12, pad=2)
    ax2.set_xlim(0, 1)

    # 3) FT top-5 bar (top class at top)
    ax3 = fig.add_subplot(gs[0, 2])
    labels_ft = [classnames[i] if i < len(classnames) else str(i) for i in ft_idx]
    y_pos_ft = np.arange(len(ft_vals))
    ax3.barh(y_pos_ft, ft_vals, color='#ff7f0e')
    ax3.set_yticks(y_pos_ft)
    ax3.set_yticklabels(labels_ft, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('FT prob')
    ax3.set_title('Finetuned Top-5', fontsize=12, pad=2)
    ax3.set_xlim(0, 1)

    # 4) GradCAM composite (from compare_outputs)
    ax4 = fig.add_subplot(gs[0, 3])
    try:
        gradcam_img = Image.open(gradcam_image_path).convert('RGB')
        ax4.imshow(gradcam_img)
        ax4.set_title('Grad-CAMs (composite)', fontsize=12, pad=2)
    except Exception as e:
        ax4.text(0.5, 0.5, f'Missing: {os.path.basename(gradcam_image_path)}', ha='center', va='center')
    ax4.axis('off')

    # Compact layout with constrained_layout
    if title:
        fig.suptitle(title, y=1.03, fontsize=12)
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    plt.savefig(fig_out)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize samples where ZS and FT predictions differ with bars and Grad-CAMs')
    parser.add_argument('--csv', type=str, required=True, help='Path to scores.csv produced by compare script')
    parser.add_argument('--compare-dir', type=str, required=True, help='Directory containing Grad-CAM composite images (compare_outputs)')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name prefix used in Grad-CAM filenames (e.g., ImageNet)')
    parser.add_argument('--classnames-source', type=str, default='openai', help='Classname source for imagenet classes (e.g., openai)')
    parser.add_argument('--out-dir', type=str, default='./diff_visualizations')
    parser.add_argument('--max-samples', type=int, default=100)

    args = parser.parse_args()

    classnames = get_classnames(args.classnames_source)
    records = read_scores(args.csv)

    # Filter to those with differing predictions
    diffs = [r for r in records if r['pred_zs'] != r['pred_ft']]
    print(f'Found {len(diffs)} differing predictions')

    count = 0
    for r in diffs:
        if count >= args.max_samples:
            break
        gidx = r['global_index']
        gt = r['gt']
        pred_zs = r['pred_zs']
        pred_ft = r['pred_ft']

        # Grad-CAM filename convention from compare script
        gradcam_name = f"{args.dataset}_idx{gidx}_gt{gt}_zs{pred_zs}_ft{pred_ft}.png"
        gradcam_path = os.path.join(args.compare_dir, gradcam_name)

        title = f"idx={gidx} | GT={classnames[gt] if gt < len(classnames) else gt} | ZS={classnames[pred_zs] if pred_zs < len(classnames) else pred_zs} | FT={classnames[pred_ft] if pred_ft < len(classnames) else pred_ft}"
        out_path = os.path.join(args.out_dir, f"diff_{gidx}.png")
        plot_sample(out_path, r['image_path'], r['zs_probs'], r['ft_probs'], classnames, gradcam_path, title=title)

        count += 1

    print(f'Saved {count} visualizations to {args.out_dir}')


if __name__ == '__main__':
    main()


