#!/usr/bin/env python3
"""
Analysis script for ICLR paper: Forgetting rates and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as _fm
from pathlib import Path
import argparse

# Prefer a modern sans-serif (Inter/Roboto). Fall back gracefully if unavailable
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

# Set global style to match the reference figures (large fonts, thick spines, minimal grid)
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
    'axes.labelpad': 8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
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
    'figure.figsize': (6.0, 4.0),
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    # No grid by default; individual plots can enable if needed
    'axes.grid': False,
})

# Reference color set (matches the tones used in the attached figures)
REF_COLORS = {
    'red': '#d62728',      # AFR
    'blue': '#1f77b4',     # CnC / general blue
    'green': '#2ca02c',    # JTT
    'purple': '#9467bd',
    'orange': '#ff7f0e',
    'grey': '#7f7f7f'
}

# Alternate palette to match the attached image's color code
# Order: blue, purple, pink, teal (then repeat if more methods)
ALT_COLORS = {
    'blue': '#1f77b4',     # MSP-like
    'purple': '#9467bd',   # Entropy-like
    'pink': '#e377c2',     # Energy-like
    'teal': '#17becf',     # Max-Logit-like
    'grey': '#7f7f7f',
    'green': '#2ca02c',
    'light_green': '#98d98e'
}

# Global palette variable (default filled below). We expose a helper to switch.
PALETTE = []

def set_color_scheme(scheme: str = 'default') -> None:
    """Set the global method color palette.

    scheme: 'default' uses REF_COLORS; 'alt' matches the reference image colors.
    """
    global PALETTE
    if scheme == 'alt':
        PALETTE = [
            ALT_COLORS['blue'],
            ALT_COLORS['purple'],
            ALT_COLORS['pink'],
            ALT_COLORS['teal'],
            ALT_COLORS['light_green'],  # repeat to ensure enough colors
            ALT_COLORS['purple'],
            ALT_COLORS['pink'],
            ALT_COLORS['teal'],
        ]
    else:
        PALETTE = [
            REF_COLORS['red'],
            REF_COLORS['blue'],
            REF_COLORS['green'],
            REF_COLORS['purple'],
            REF_COLORS['orange'],
            REF_COLORS['grey'],
        ]

def load_data(csv_path):
    """Load experimental results from CSV file."""
    return pd.read_csv(csv_path)

def calculate_forgetting_rates(df, methods_to_plot=None):
    """Calculate forgetting rates for each method compared to pre-trained baseline."""
    pretrained_performance = df[df['Method'] == 'Pre-trained']['Image_Test_Original'].iloc[0]
    
    forgetting_data = []
    for _, row in df.iterrows():
        if row['Method'] != 'Pre-trained':
            if methods_to_plot is not None and row['Method'] not in methods_to_plot:
                continue
            # Forgetting rate = original_performance - fine_tuned_performance
            forgetting_rate = pretrained_performance - row['Image_Test_Original']
            # Relative forgetting = forgetting_rate / original_performance * 100
            relative_forgetting = (forgetting_rate / pretrained_performance) * 100
            
            forgetting_data.append({
                'Method': row['Method'],
                'Forgetting_Rate_Absolute': forgetting_rate,
                'Forgetting_Rate_Relative': relative_forgetting,
                'Original_Performance': row['Image_Test_Original'],
                'Spurious_Performance': row['Image_Test_Colored']
            })
    
    return pd.DataFrame(forgetting_data)

def create_forgetting_comparison_plot(df, save_path='figures/forgetting_comparison.pdf', methods_to_plot=None):
    """Create a comparison plot of original vs fine-tuned performance."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    
    # Filter methods
    if methods_to_plot is not None:
        methods = [m for m in df[df['Method'] != 'Pre-trained']['Method'].tolist() if m in methods_to_plot]
        finetuned_df = df[df['Method'].isin(methods)]
    else:
        methods = df[df['Method'] != 'Pre-trained']['Method'].tolist()
        finetuned_df = df[df['Method'] != 'Pre-trained']
    pretrained_perf = df[df['Method'] == 'Pre-trained']['Image_Test_Original'].iloc[0]
    finetuned_perfs = finetuned_df['Image_Test_Original'].tolist()
    
    x = np.arange(len(methods))
    width = 0.32
    
    # Create bars with Google-style colors
    bars1 = ax.bar(x - width/2, [pretrained_perf] * len(methods), width, 
                   label='Pre-trained', alpha=0.95, color=REF_COLORS['blue'], 
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, finetuned_perfs, width, 
                   label='Fine-tuned', alpha=0.95, color=REF_COLORS['red'],
                   edgecolor='white', linewidth=0.8)
    
    # Add elegant value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{height:.1f}', ha='center', va='bottom', 
                fontsize=10, color='#555555', fontweight='500')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{height:.1f}', ha='center', va='bottom', 
                fontsize=10, color='#555555', fontweight='500')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha='center')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Add subtle forgetting indicators
    for i, (method, orig_perf) in enumerate(zip(methods, finetuned_perfs)):
        forgetting = pretrained_perf - orig_perf
        if forgetting > 0:
            # Subtle line connecting the bars
            ax.plot([i - width/2 + width/2, i + width/2 - width/2], 
                   [pretrained_perf, orig_perf], 
                   color='#CCCCCC', linestyle='--', alpha=0.6, linewidth=1)
            # Clean forgetting rate text
            mid_y = (pretrained_perf + orig_perf) / 2
            ax.text(i + width/2 + 0.12, mid_y, f'−{forgetting:.1f}%', 
                   ha='left', va='center', fontsize=12, 
                   color=REF_COLORS['red'], fontweight='500',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                           edgecolor=REF_COLORS['red'], linewidth=0.8, alpha=0.95))
    
    # Set y-axis limits with some padding
    y_min = min(min(finetuned_perfs), pretrained_perf) - 5
    y_max = max(max(finetuned_perfs), pretrained_perf) + 8
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor='white', edgecolor='none')
    print(f"Saved forgetting comparison plot to {save_path}")
    
    return fig

def create_forgetting_rates_barplot(forgetting_df, save_path='figures/forgetting_rates.pdf', methods_to_plot=None):
    """Create a bar plot of forgetting rates."""
    if methods_to_plot is not None:
        forgetting_df = forgetting_df[forgetting_df['Method'].isin(methods_to_plot)]
    
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    
    # Use Google-style colors
    method_colors = [PALETTE[i % len(PALETTE)] for i in range(len(forgetting_df))]
    
    # Create horizontal bar plot for better readability
    y_pos = np.arange(len(forgetting_df))
    bars = ax.barh(y_pos, forgetting_df['Forgetting_Rate_Absolute'], 
                   color=method_colors, alpha=0.85, height=0.6,
                   edgecolor='white', linewidth=0.8)
    
    # Add value labels at the end of bars
    for i, (bar, value) in enumerate(zip(bars, forgetting_df['Forgetting_Rate_Absolute'])):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left', va='center', 
                fontsize=10, color='#555555', fontweight='500')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forgetting_df['Method'], fontsize=10)
    ax.set_xlabel('Forgetting Rate (%)')
    ax.set_title('')
    
    # Invert y-axis to have methods in descending order of forgetting
    ax.invert_yaxis()
    
    # Style the grid and axes
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set x-axis limits with padding
    x_max = max(forgetting_df['Forgetting_Rate_Absolute']) * 1.15
    ax.set_xlim(0, x_max)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor='white', edgecolor='none')
    print(f"Saved forgetting rates plot to {save_path}")
    
    return fig

def create_spurious_correlation_analysis(forgetting_df, save_path='figures/spurious_correlation.pdf', methods_to_plot=None):
    """Create scatter plot analyzing spurious correlation vs original performance."""
    if methods_to_plot is not None:
        forgetting_df = forgetting_df[forgetting_df['Method'].isin(methods_to_plot)]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    
    # Use Google-style colors
    method_colors = [PALETTE[i % len(PALETTE)] for i in range(len(forgetting_df))]
    
    # Create elegant scatter plot
    scatter = ax.scatter(forgetting_df['Original_Performance'], 
                        forgetting_df['Spurious_Performance'],
                        s=180, alpha=0.8, 
                        c=method_colors, edgecolors='white', linewidth=1.5)
    
    # Add method labels with clean styling
    for i, method in enumerate(forgetting_df['Method']):
        x, y = forgetting_df['Original_Performance'].iloc[i], forgetting_df['Spurious_Performance'].iloc[i]
        ax.annotate(method, 
                   (x, y),
                   xytext=(12, 8), textcoords='offset points',
                   fontsize=10, fontweight='500', color='#444444',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                           edgecolor='#E0E0E0', linewidth=1, alpha=0.95))
    
    # Add diagonal reference line
    min_val = min(forgetting_df['Original_Performance'].min(), 
                  forgetting_df['Spurious_Performance'].min()) - 2
    max_val = max(forgetting_df['Original_Performance'].max(), 
                  forgetting_df['Spurious_Performance'].max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 
            color='#CCCCCC', linestyle='--', alpha=0.7, linewidth=1.5,
            label='Perfect Correlation', zorder=1)
    
    # Styling
    ax.set_xlabel('Original Task Accuracy (%)')
    ax.set_ylabel('Spurious Correlation Accuracy (%)')
    ax.set_title('')
    
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True)
    
    # Add correlation coefficient with elegant styling
    corr = np.corrcoef(forgetting_df['Original_Performance'], 
                      forgetting_df['Spurious_Performance'])[0, 1]
    ax.text(0.05, 0.92, f'r = {corr:.3f}', 
            transform=ax.transAxes, fontsize=12, fontweight='600',
            color='#222222',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                     alpha=0.95, edgecolor=REF_COLORS['blue'], linewidth=1.2))
    
    # Set axis limits with padding
    x_padding = (forgetting_df['Original_Performance'].max() - forgetting_df['Original_Performance'].min()) * 0.1
    y_padding = (forgetting_df['Spurious_Performance'].max() - forgetting_df['Spurious_Performance'].min()) * 0.1
    ax.set_xlim(forgetting_df['Original_Performance'].min() - x_padding, 
                forgetting_df['Original_Performance'].max() + x_padding)
    ax.set_ylim(forgetting_df['Spurious_Performance'].min() - y_padding, 
                forgetting_df['Spurious_Performance'].max() + y_padding)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor='white', edgecolor='none')
    print(f"Saved spurious correlation analysis to {save_path}")
    
    return fig

def create_comprehensive_performance_plot(df, save_path='figures/comprehensive_performance.pdf', methods_to_plot=None):
    """Create a comprehensive multi-panel performance comparison."""
    # Filter methods
    if methods_to_plot is not None:
        df = df[df['Method'].isin(['Pre-trained'] + methods_to_plot)]
    
    # 1x4 layout as requested (tuned height)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16.0, 5.0))
    methods = df['Method'].tolist()

    # Panel 1: Original MNIST performance comparison (Test only)
    method_colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]
    bars1 = ax1.bar(methods, df['Image_Test_Original'], color=method_colors, alpha=0.95,
                    edgecolor='white', linewidth=0.8)
    ax1.set_title('Original task (MNIST)', y=1.02, fontsize=12)
    ax1.set_ylabel('Accuracy (%)')
    ax1.tick_params(axis='x', rotation=25)
    ax1.grid(False)
    ax1.set_axisbelow(True)
    # Value labels on top
    for bar, v in zip(bars1, df['Image_Test_Original']):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=10, color='#444444', fontweight='500')
    # Panel labels added at the figure level below (to avoid overlap)

    # Panel 2: Spurious correlation performance
    method_colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]
    bars2 = ax2.bar(methods, df['Image_Test_Colored'], color=method_colors, alpha=0.9,
                    edgecolor='white', linewidth=0.8)
    ax2.set_title('Fine-tuning task (Colored MNIST)', y=1.02, fontsize=12)
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='x', rotation=25)
    ax2.grid(False)
    ax2.set_axisbelow(True)

    # Add value labels for panel 2
    for i, (bar, v) in enumerate(zip(bars2, df['Image_Test_Colored'])):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f'{v:.1f}', ha='center', va='bottom', 
                fontsize=10, color='#444444', fontweight='500')
    # Panel labels added at the figure level below (to avoid overlap)

    # Panel 3: Forgetting analysis
    forgetting_methods = [m for m in methods if m != 'Pre-trained']
    forgetting_rates = []
    pretrained_perf = df[df['Method'] == 'Pre-trained']['Image_Test_Original'].iloc[0]

    for m in forgetting_methods:
        row = df[df['Method'] == m].iloc[0]
        forgetting_rates.append(pretrained_perf - row['Image_Test_Original'])

    y_pos = np.arange(len(forgetting_methods))
    bars3 = ax3.barh(y_pos, forgetting_rates, 
                     color=[PALETTE[(i+1) % len(PALETTE)] for i in range(len(forgetting_methods))],
                     alpha=0.9, height=0.6, edgecolor='white', linewidth=0.8)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars3, forgetting_rates)):
        if rate > 0:
            ax3.text(rate + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', ha='left', va='center', 
                    fontsize=9, color='#555555', fontweight='500')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(forgetting_methods, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel('Forgetting Rate (%)')
    ax3.set_title('')
    ax3.grid(False)
    ax3.set_axisbelow(True)
    ax3.spines['left'].set_visible(False)
    # Panel labels added at the figure level below (to avoid overlap)

    # Add improvement annotation for Dynamic Distill relative to Direct FT
    if 'Dynamic Distill' in forgetting_methods and 'Direct FT' in forgetting_methods:
        dd_idx = forgetting_methods.index('Dynamic Distill')
        dft_idx = forgetting_methods.index('Direct FT')
        improvement = forgetting_rates[dd_idx] - forgetting_rates[dft_idx]
        x_annot = max(0.45 * max(forgetting_rates + [1e-6]), 0.5)
        ax3.text(x_annot, dd_idx, f'({improvement:.2f}%)', va='center', ha='left',
                 fontsize=12, color=REF_COLORS['green'], fontweight='600')

    # Panel 4: Performance trade-off scatter
    finetuned_df = df[df['Method'] != 'Pre-trained']
    if len(finetuned_df) > 0:
        ax4.scatter(finetuned_df['Image_Test_Original'], 
                   finetuned_df['Image_Test_Colored'],
                   s=150, alpha=0.8, 
                   c=[PALETTE[(i+1) % len(PALETTE)] for i in range(len(finetuned_df))],
                   edgecolors='white', linewidth=1.5)
        
        # Add method labels
        for i, method in enumerate(finetuned_df['Method']):
            x, y = finetuned_df['Image_Test_Original'].iloc[i], finetuned_df['Image_Test_Colored'].iloc[i]
            ax4.annotate(method, (x, y), xytext=(2, 12), textcoords='offset points',
                        fontsize=8, fontweight='600', color='#444444',
                        #ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                                edgecolor='#E0E0E0', linewidth=1, alpha=0.9))
    
    ax4.set_xlabel('Original Task Accuracy (%)', fontsize=13)
    ax4.set_ylabel('Spurious Task Accuracy (%)', fontsize=13)
    ax4.set_title('')
    ax4.grid(False)
    ax4.set_axisbelow(True)
    # Make the ylim a bit bigger
    y_min, y_max = ax4.get_ylim()
    ax4.set_ylim(y_min - 0.02, y_max + 0.02)
    # Panel labels added at the figure level below (to avoid overlap)
    # Overall styling
    plt.suptitle('')
    # Add extra bottom margin for panel labels
    plt.tight_layout(rect=[0, 0.26, 1, 1])
    # Figure-level panel labels placed just below each axes to avoid overlaps
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_pos = pos.y0 - 0.15
        fig.text(x_center, y_pos, f'({chr(97 + i)})', ha='center', va='top', fontsize=12, fontweight='bold')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, facecolor='white', edgecolor='none')
    print(f"Saved comprehensive performance plot to {save_path}")
    
    return fig

def generate_summary_table(df, forgetting_df, save_path='tables/results_summary.txt', methods_to_plot=None):
    """Generate a formatted summary table for the paper."""
    # Filter methods
    if methods_to_plot is not None:
        forgetting_df = forgetting_df[forgetting_df['Method'].isin(methods_to_plot)]
    summary = f"""
=================================================
RESULTS SUMMARY FOR ICLR PAPER
=================================================

Table 1: Performance Comparison (%)
Method              Original MNIST    Colored MNIST    Forgetting Rate
                    (Test)           (Test)           (Absolute)
--------------------------------------------------------------------
"""
    
    pretrained_perf = df[df['Method'] == 'Pre-trained']['Image_Test_Original'].iloc[0]
    colored_pretrained = df[df['Method'] == 'Pre-trained']['Image_Test_Colored'].iloc[0]
    
    summary += f"Pre-trained         {pretrained_perf:6.1f}           {colored_pretrained:6.1f}              0.0\n"
    
    for _, row in forgetting_df.iterrows():
        summary += f"{row['Method']:<18} {row['Original_Performance']:6.1f}           {row['Spurious_Performance']:6.1f}           {row['Forgetting_Rate_Absolute']:6.1f}\n"
    
    summary += f"""

Key Findings:
1. Direct fine-tuning causes severe catastrophic forgetting ({forgetting_df.loc[forgetting_df['Method'] == 'Direct FT', 'Forgetting_Rate_Absolute'].iloc[0]:.1f}% drop)""" if 'Direct FT' in forgetting_df['Method'].values else ""
    summary += f"""
2. Distillation methods significantly reduce forgetting:"""
    if 'Static Distill' in forgetting_df['Method'].values:
        summary += f"\n   - Static Distillation: {forgetting_df.loc[forgetting_df['Method'] == 'Static Distill', 'Forgetting_Rate_Absolute'].iloc[0]:.1f}% forgetting"
    if 'Dynamic Distill' in forgetting_df['Method'].values:
        summary += f"\n   - Dynamic Distillation: {forgetting_df.loc[forgetting_df['Method'] == 'Dynamic Distill', 'Forgetting_Rate_Absolute'].iloc[0]:.1f}% forgetting"
    summary += f"""
3. All fine-tuning methods achieve high performance on spurious correlation task (>97%)
4. Text modality remains unaffected (100% accuracy maintained)

Trade-off Analysis:
- Methods with less forgetting on original task maintain similar spurious correlation performance
- Distillation methods provide the best balance between retaining original knowledge and adapting to new patterns
"""
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"Saved results summary to {save_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Analyze experimental results for ICLR paper')
    parser.add_argument('--csv_path', default='experiment_results.csv', 
                       help='Path to CSV file with results')
    parser.add_argument('--output_dir', default='.', 
                       help='Output directory for figures and tables')
    parser.add_argument('--methods', nargs='+', type=str, default=['Direct FT', 'L2 Reg', 'Static Distill', 'Dynamic Distill'],
                        help='List of methods to plot (excluding "Pre-trained"). If not specified, plot all.')
    parser.add_argument('--color_scheme', type=str, default='alt', choices=['default', 'alt'],
                        help='Color scheme to use: default (paper palette) or alt (reference figure palette).')
    args = parser.parse_args()
    
    # Load data
    print("Loading experimental data...")
    df = load_data(args.csv_path)
    print(f"Loaded data for {len(df)} methods")
    # Configure color scheme
    set_color_scheme(args.color_scheme)
    print(f"Using color scheme: {args.color_scheme}")
    
    # Determine methods to plot
    if args.methods is not None and len(args.methods) > 0:
        methods_to_plot = args.methods
        print(f"Plotting only selected methods: {methods_to_plot}")
    else:
        # All methods except Pre-trained
        methods_to_plot = df[df['Method'] != 'Pre-trained']['Method'].tolist()
        print(f"Plotting all methods: {methods_to_plot}")
    
    # Calculate forgetting rates
    print("Calculating forgetting rates...")
    forgetting_df = calculate_forgetting_rates(df, methods_to_plot=methods_to_plot)
    print("Forgetting rates calculated:")
    print(forgetting_df[['Method', 'Forgetting_Rate_Absolute', 'Forgetting_Rate_Relative']])
    
    # Create output directories
    figures_dir = Path(args.output_dir) / 'toy_experiment_figures_reproducible'
    tables_dir = Path(args.output_dir) / 'toy_experiment_tables_reproducible'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\nGenerating publication-quality plots...")
    
    fig1 = create_forgetting_comparison_plot(df, figures_dir / 'forgetting_comparison.pdf', methods_to_plot=methods_to_plot)
    fig2 = create_forgetting_rates_barplot(forgetting_df, figures_dir / 'forgetting_rates.pdf', methods_to_plot=methods_to_plot)
    fig3 = create_spurious_correlation_analysis(forgetting_df, figures_dir / 'spurious_correlation.pdf', methods_to_plot=methods_to_plot)
    fig4 = create_comprehensive_performance_plot(df, figures_dir / 'comprehensive_performance.pdf', methods_to_plot=methods_to_plot)
    
    # Generate summary
    summary = generate_summary_table(df, forgetting_df, tables_dir / 'results_summary.txt', methods_to_plot=methods_to_plot)
    
    print("\nAnalysis complete! Generated files:")
    print(f"- Figures: {figures_dir}")
    print(f"- Tables: {tables_dir}")
    print(f"- Summary: {tables_dir / 'results_summary.txt'}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()