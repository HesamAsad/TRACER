import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict

# --- Configuration ---
CSV_PATH = 'experiment_1_detailed.csv'
OUTPUT_DIR = 'visualizations_all_ood'
DPI = 300

# --- Plotting Style ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("deep", 5)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw data for easier plotting."""
    
    # Create a simplified 'method_family' column for grouping
    def get_method_family(method_name):
        if 'L2-SP' in method_name:
            return 'L2-SP'
        if 'SD-Static' in method_name:
            return 'SD-Static'
        if 'SD-BMA' in method_name:
            return 'SD-BMA'
        return 'Direct FT'
        
    df['method_family'] = df['method'].apply(get_method_family)
    
    # Recalculate fraction_orthogonal for consistency, as some values look odd
    # ||ΔW||_F = sqrt(||ΔW P_I||_F^2 + ||ΔW(I-P_I)||_F^2)
    norm_total = np.sqrt(df['norm_in_subspace']**2 + df['norm_orthogonal']**2)
    # Avoid division by zero
    df['fraction_orthogonal_recalc'] = np.divide(df['norm_orthogonal'], norm_total, 
                                                out=np.zeros_like(df['norm_orthogonal']), 
                                                where=norm_total!=0)
    
    # Ensure method order for consistent plotting
    method_order = ['Direct FT', 'L2-SP', 'SD-Static', 'SD-BMA']
    df['method_family'] = pd.Categorical(df['method_family'], categories=method_order, ordered=True)
    
    print("Data preprocessed successfully.")
    
    return df

# ==============================================================================
# METRIC-DEPENDENT PLOTS (will be generated for each OOD type)
# ==============================================================================

def plot_1_robustness_vs_coverage(df: pd.DataFrame, color_map: Dict, ood_metric: str):
    """
    PLOT 1: OOD accuracy vs. Subspace Coverage (K_rot).
    Shows the main performance result for a specific OOD scenario.
    """
    metric_labels = {
        'ood_both': ('OOD-Both', 'Both Color & Pattern Randomized'),
        'ood_color': ('OOD-Color', 'Color Randomized'),
        'ood_pattern': ('OOD-Pattern', 'Pattern Randomized')
    }
    label, subtitle = metric_labels[ood_metric]

    plt.figure(figsize=(8, 5))
    
    ax = sns.lineplot(
        data=df,
        x='K_rot',
        y=ood_metric,
        hue='method_family',
        style='method_family',
        markers=True,
        dashes=False,
        errorbar='ci',
        palette=color_map,
        linewidth=2.5
    )
    
    ax.set_title(f'Robustness vs. Subspace Coverage ({subtitle})', fontsize=16, pad=20)
    ax.set_xlabel('Subspace Coverage ($K_{rot}$)', fontsize=12)
    ax.set_ylabel(f'{label} Accuracy', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['K_rot'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.4, 1.0)
    
    plt.legend(title='Method', loc='lower right')
    plt.tight_layout()
    
    filename = f'1_robustness_vs_coverage_{ood_metric}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)
    print(f"Generated plot: {filename}")
    plt.close()

def plot_5_best_method_summary(df: pd.DataFrame, color_map: Dict, ood_metric: str):
    """
    PLOT 5: Summary comparing Direct FT against the best version of each regularized method.
    The "best" version is selected based on the specified ood_metric.
    """
    metric_labels = {
        'ood_both': 'OOD-Both',
        'ood_color': 'OOD-Color',
        'ood_pattern': 'OOD-Pattern'
    }
    label = metric_labels[ood_metric]

    # Find the best method (full name) for each family based on the specified ood_metric
    best_methods_idx = df.groupby(['method_family', 'method'])[ood_metric].mean().groupby('method_family').idxmax()
    best_method_names = [name for family, name in best_methods_idx] + ['Direct FT']
    
    df_best = df[df['method'].isin(best_method_names)].copy()
    df_best['method_short'] = df_best['method'].apply(lambda x: x.split(' (')[0])
    
    # Aggregate results for these best methods, always showing ID, OS, and the target OOD metric
    columns_to_agg = ['id_test', ood_metric, 'orthogonal_subspace']
    df_agg = df_best.groupby('method_short')[columns_to_agg].mean().reset_index()
    
    # Melt for plotting
    df_melted = df_agg.melt(id_vars='method_short', var_name='Metric', value_name='Accuracy')
    
    # Create the mapping for the legend labels dynamically
    metric_map = {'id_test': 'ID', 'orthogonal_subspace': 'OS', ood_metric: label}
    df_melted['Metric'] = df_melted['Metric'].map(metric_map)
    
    # Ensure consistent ordering in the plot
    metric_order = ['ID', label, 'OS']
    df_melted['Metric'] = pd.Categorical(df_melted['Metric'], categories=metric_order, ordered=True)
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=df_melted,
        x='method_short',
        y='Accuracy',
        hue='Metric',
        palette=sns.color_palette("viridis", 3)
    )
    
    ax.set_title(f'Performance Summary (Best Methods for {label})', fontsize=16, pad=20)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0.0, 1.1)
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9)

    plt.legend(title='Accuracy Metric')
    plt.tight_layout()
    
    filename = f'5_best_method_summary_{ood_metric}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)
    print(f"Generated plot: {filename}")
    plt.close()

# ==============================================================================
# METRIC-INDEPENDENT PLOTS (generated only once)
# ==============================================================================

def plot_2_os_accuracy_vs_coverage(df: pd.DataFrame, color_map: Dict):
    """
    PLOT 2: Orthogonal Subspace (OS) Accuracy vs. Subspace Coverage (K_rot).
    Shows that regularized methods are better at preserving knowledge outside the
    finetuning data subspace.
    """
    plt.figure(figsize=(8, 5))
    
    ax = sns.lineplot(
        data=df,
        x='K_rot',
        y='orthogonal_subspace',
        hue='method_family',
        style='method_family',
        markers=True,
        dashes=False,
        errorbar='ci',
        palette=color_map,
        linewidth=2.5
    )

    ax.set_title('Regularization Preserves Knowledge Outside Finetuning Subspace', fontsize=16, pad=20)
    ax.set_xlabel('Subspace Coverage ($K_{rot}$)', fontsize=12)
    ax.set_ylabel('Orthogonal Subspace (OS) Accuracy', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['K_rot'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.4, 1.0)
    
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_os_accuracy_vs_coverage.png'), dpi=DPI)
    print("Generated plot: 2_os_accuracy_vs_coverage.png")
    plt.close()

def plot_3_geometric_decomposition(df: pd.DataFrame, color_map: Dict):
    """
    PLOT 3: Geometric Decomposition of Weight Updates.
    Directly validates the paper's theory, showing the magnitudes of 
    in-subspace vs. orthogonal updates.
    """
    df_agg = df.groupby('method_family')[['norm_in_subspace', 'norm_orthogonal']].mean().reset_index()
    df_melted = df_agg.melt(id_vars='method_family', var_name='Component', value_name='Frobenius Norm')
    
    df_melted['Component'] = df_melted['Component'].map({
        'norm_in_subspace': r'In-Subspace ($\| \Delta W \mathcal{P}_I \|_F$)',
        'norm_orthogonal': r'Orthogonal ($\| \Delta W (\mathbf{I}-\mathcal{P}_I) \|_F$)'
    })

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=df_melted,
        x='method_family',
        y='Frobenius Norm',
        hue='Component',
        palette='pastel'
    )
    
    ax.set_title('Geometric Decomposition of Vision Encoder Updates', fontsize=16, pad=20)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Average Frobenius Norm', fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    
    plt.legend(title='Update Component', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_geometric_decomposition.png'), dpi=DPI)
    print("Generated plot: 3_geometric_decomposition.png")
    plt.close()

def plot_4_update_tradeoff_scatter(df: pd.DataFrame, color_map: Dict):
    """
    PLOT 4: In-Subspace vs. Orthogonal Update Norms (Scatter).
    Visualizes the trade-off space, showing how different methods cluster
    in terms of their update geometry.
    """
    plt.figure(figsize=(7, 7))
    
    ax = sns.scatterplot(
        data=df,
        x='norm_orthogonal',
        y='norm_in_subspace',
        hue='method_family',
        palette=color_map,
        alpha=0.6,
        s=50,
        edgecolor='w'
    )
    
    ax.set_title('Adaptation vs. Forgetting Trade-off Space', fontsize=16, pad=20)
    ax.set_xlabel(r'Orthogonal Update Norm (Forgetting)', fontsize=12)
    ax.set_ylabel(r'In-Subspace Update Norm (Adaptation)', fontsize=12)
    
    # Highlight Direct FT's high forgetting
    ax.text(2.5, 3.5, 'Direct FT:\nHigh Forgetting', color=color_map['Direct FT'], fontsize=11, weight='bold', ha='center')
    ax.text(0.5, 0.6, 'Regularized Methods:\nConstrained Forgetting', color='grey', fontsize=11, weight='bold', ha='center')

    plt.legend(title='Method', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_update_tradeoff_scatter.png'), dpi=DPI)
    print("Generated plot: 4_update_tradeoff_scatter.png")
    plt.close()

def main():
    """Main function to run the visualization generation."""
    if not os.path.exists(CSV_PATH):
        print(f"Error: The file '{CSV_PATH}' was not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)
    df_processed = preprocess_data(df)
    
    method_families = df_processed['method_family'].cat.categories
    color_map = {family: color for family, color in zip(method_families, palette)}
    
    # --- Generate Metric-INDEPENDENT Plots ---
    print("\n--- Generating Metric-Independent Plots (1 copy each) ---")
    plot_2_os_accuracy_vs_coverage(df_processed, color_map)
    plot_3_geometric_decomposition(df_processed, color_map)
    plot_4_update_tradeoff_scatter(df_processed, color_map)

    # --- Generate Metric-DEPENDENT Plots for each OOD scenario ---
    ood_metrics = ['ood_both', 'ood_color', 'ood_pattern']
    for metric in ood_metrics:
        print(f"\n--- Generating Plots for Metric: {metric.upper()} ---")
        plot_1_robustness_vs_coverage(df_processed, color_map, ood_metric=metric)
        plot_5_best_method_summary(df_processed, color_map, ood_metric=metric)
    
    print(f"\nAll visualizations have been saved to the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    main()