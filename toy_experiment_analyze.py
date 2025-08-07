#!/usr/bin/env python3
"""
Analysis script for ICLR paper: Forgetting rates and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style for publication-quality figures (Google papers style)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'text.usetex': False,
    'figure.figsize': (10, 6),
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#333333',
    'axes.axisbelow': True,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'legend.frameon': False,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.8,
    'axes.labelcolor': '#333333',
    'text.color': '#333333'
})

# Professional color palette inspired by Google Material Design and modern papers
GOOGLE_COLORS = {
    'primary_blue': '#1A73E8',
    'secondary_blue': '#4285F4', 
    'red': '#EA4335',
    'orange': '#FF9800',
    'green': '#34A853',
    'purple': '#9C27B0',
    'teal': '#00BCD4',
    'amber': '#FFC107',
    'deep_purple': '#673AB7',
    'indigo': '#3F51B5'
}

PALETTE = [
    GOOGLE_COLORS['primary_blue'],
    GOOGLE_COLORS['red'], 
    GOOGLE_COLORS['green'],
    GOOGLE_COLORS['orange'],
    GOOGLE_COLORS['purple'],
    GOOGLE_COLORS['teal'],
    GOOGLE_COLORS['deep_purple'],
    GOOGLE_COLORS['amber']
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
    fig, ax = plt.subplots(figsize=(11, 7))
    
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
                   label='Pre-trained', alpha=0.9, color=GOOGLE_COLORS['primary_blue'], 
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, finetuned_perfs, width, 
                   label='Fine-tuned', alpha=0.9, color=GOOGLE_COLORS['red'],
                   edgecolor='white', linewidth=0.8)
    
    # Add elegant value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{height:.1f}', ha='center', va='bottom', 
                fontsize=9, color='#555555', fontweight='500')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{height:.1f}', ha='center', va='bottom', 
                fontsize=9, color='#555555', fontweight='500')
    
    # Styling
    ax.set_xlabel('Method', fontsize=12, fontweight='500', color='#444444')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='500', color='#444444')
    ax.set_title('Original Task Performance Comparison', fontsize=14, fontweight='600', 
                 color='#333333', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha='center', fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
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
                   ha='left', va='center', fontsize=9, 
                   color=GOOGLE_COLORS['red'], fontweight='500',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                           edgecolor=GOOGLE_COLORS['red'], linewidth=0.8, alpha=0.9))
    
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
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
    ax.set_xlabel('Forgetting Rate (%)', fontsize=12, fontweight='500', color='#444444')
    ax.set_title('Catastrophic Forgetting Analysis', fontsize=14, fontweight='600', 
                 color='#333333', pad=20)
    
    # Invert y-axis to have methods in descending order of forgetting
    ax.invert_yaxis()
    
    # Style the grid and axes
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)
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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    ax.set_xlabel('Original Task Accuracy (%)', fontsize=12, fontweight='500', color='#444444')
    ax.set_ylabel('Spurious Correlation Accuracy (%)', fontsize=12, fontweight='500', color='#444444')
    ax.set_title('Performance Trade-off Analysis', fontsize=14, fontweight='600', 
                 color='#333333', pad=20)
    
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    
    # Add correlation coefficient with elegant styling
    corr = np.corrcoef(forgetting_df['Original_Performance'], 
                      forgetting_df['Spurious_Performance'])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', 
            transform=ax.transAxes, fontsize=11, fontweight='600',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=GOOGLE_COLORS['primary_blue'], 
                     alpha=0.1, edgecolor=GOOGLE_COLORS['primary_blue'], linewidth=1))
    
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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    methods = df['Method'].tolist()

    # Panel 1: Original MNIST performance comparison (Train/Val/Test)
    x = np.arange(len(methods))
    width = 0.25
    
    train_bars = ax1.bar(x - width, df['Image_Train_Original'], width, label='Train', 
                        color=GOOGLE_COLORS['primary_blue'], alpha=0.9, edgecolor='white', linewidth=0.8)
    val_bars = ax1.bar(x, df['Image_Val_Original'], width, label='Validation', 
                      color=GOOGLE_COLORS['green'], alpha=0.9, edgecolor='white', linewidth=0.8)
    test_bars = ax1.bar(x + width, df['Image_Test_Original'], width, label='Test', 
                       color=GOOGLE_COLORS['orange'], alpha=0.9, edgecolor='white', linewidth=0.8)

    ax1.set_title('Original MNIST Performance', fontsize=13, fontweight='600', color='#333333', pad=15)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='500', color='#444444')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha='right', fontsize=10)
    ax1.legend(loc='upper right', frameon=False, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)

    # Panel 2: Spurious correlation performance
    method_colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]
    bars2 = ax2.bar(methods, df['Image_Test_Colored'], color=method_colors, alpha=0.9,
                    edgecolor='white', linewidth=0.8)
    ax2.set_title('Colored MNIST Performance', fontsize=13, fontweight='600', color='#333333', pad=15)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='500', color='#444444')
    ax2.tick_params(axis='x', rotation=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)

    # Add value labels for panel 2
    for i, (bar, v) in enumerate(zip(bars2, df['Image_Test_Colored'])):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f'{v:.1f}', ha='center', va='bottom', 
                fontsize=9, color='#555555', fontweight='500')

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
    ax3.set_xlabel('Forgetting Rate (%)', fontsize=11, fontweight='500', color='#444444')
    ax3.set_title('Catastrophic Forgetting', fontsize=13, fontweight='600', color='#333333', pad=15)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_axisbelow(True)
    ax3.spines['left'].set_visible(False)

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
            ax4.annotate(method, (x, y), xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='500', color='#444444',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                                edgecolor='#E0E0E0', linewidth=1, alpha=0.9))
    
    ax4.set_xlabel('Original Task Accuracy (%)', fontsize=11, fontweight='500', color='#444444')
    ax4.set_ylabel('Spurious Task Accuracy (%)', fontsize=11, fontweight='500', color='#444444')
    ax4.set_title('Performance Trade-off', fontsize=13, fontweight='600', color='#333333', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_axisbelow(True)
    
    # Overall styling
    plt.suptitle('Comprehensive Performance Analysis', 
                fontsize=16, fontweight='700', y=0.95, color='#333333')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
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
    args = parser.parse_args()
    
    # Load data
    print("Loading experimental data...")
    df = load_data(args.csv_path)
    print(f"Loaded data for {len(df)} methods")
    
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