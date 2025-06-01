import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for high-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.2
})

# Load the data
df_carot = pd.read_csv('expt_logs/ImageNet/carot/16_ep10_BS512_WD0.1_LR1e-05_D1.5_OC0.2_CF0.05_run1/stats.tsv', sep='\t', index_col=0)
df_ldreg = pd.read_csv('expt_logs/ImageNet/carot_ldreg/16_ep10_BS512_WD0.1_LR1e-05_D1.5_OC0.2_CF0.05_LDReg0.01_k64_run1/stats.tsv', sep='\t', index_col=0)
df_carot.drop(9, inplace=True, axis=0)

# Define colors for consistency
colors = {
    'CaRot': '#2E86AB',
    'LDReg': '#A23B72'
}

# Create figure directory if it doesn't exist
import os
os.makedirs('comparison_plots', exist_ok=True)

# Plot 1: Training Loss Components Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Loss Components: CaRot vs LDReg', fontsize=16, fontweight='bold')

# Total Loss
axes[0,0].plot(df_carot['epoch'], df_carot['Avg Total Loss'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[0,0].plot(df_ldreg['epoch'], df_ldreg['Avg Total Loss'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[0,0].set_title('Total Loss', fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# CLIP Loss
axes[0,1].plot(df_carot['epoch'], df_carot['Avg CLIP Loss'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[0,1].plot(df_ldreg['epoch'], df_ldreg['Avg CLIP Loss'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[0,1].set_title('CLIP Loss', fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Orthogonality Loss
axes[1,0].plot(df_carot['epoch'], df_carot['Avg Orthogonality Loss'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[1,0].plot(df_ldreg['epoch'], df_ldreg['Avg Orthogonality Loss'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[1,0].set_title('Orthogonality Loss', fontweight='bold')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Distillation Loss
axes[1,1].plot(df_carot['epoch'], df_carot['Avg Distillation Loss'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[1,1].plot(df_ldreg['epoch'], df_ldreg['Avg Distillation Loss'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[1,1].set_title('Distillation Loss', fontweight='bold')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Loss')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/01_loss_components.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: ImageNet Performance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ImageNet Performance: Accuracy vs Calibration', fontsize=16, fontweight='bold')

# Accuracy
ax1.plot(df_carot['epoch'], df_carot['ImageNet Accuracy'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax1.plot(df_ldreg['epoch'], df_ldreg['ImageNet Accuracy'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax1.set_title('ImageNet Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.75, 0.84)

# ECE (Expected Calibration Error)
ax2.plot(df_carot['epoch'], df_carot['ImageNet ECE'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax2.plot(df_ldreg['epoch'], df_ldreg['ImageNet ECE'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax2.set_title('ImageNet ECE (Lower is Better)', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Expected Calibration Error')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/02_imagenet_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Robustness Evaluation - Accuracy Across Datasets
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Robustness Evaluation: Accuracy Across ImageNet Variants', fontsize=16, fontweight='bold')

datasets = ['ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']
positions = [(0,0), (0,1), (1,0), (1,1)]

for i, (dataset, pos) in enumerate(zip(datasets, positions)):
    ax = axes[pos]
    acc_col = f'{dataset} Accuracy'
    
    ax.plot(df_carot['epoch'], df_carot[acc_col], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
    ax.plot(df_ldreg['epoch'], df_ldreg[acc_col], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
    ax.set_title(f'{dataset} Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/03_robustness_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Calibration Across Datasets
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Calibration Quality: ECE Across ImageNet Variants', fontsize=16, fontweight='bold')

for i, (dataset, pos) in enumerate(zip(datasets, positions)):
    ax = axes[pos]
    ece_col = f'{dataset} ECE'
    
    ax.plot(df_carot['epoch'], df_carot[ece_col], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
    ax.plot(df_ldreg['epoch'], df_ldreg[ece_col], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
    ax.set_title(f'{dataset} ECE', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Expected Calibration Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/04_calibration_quality.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Final Performance Comparison (Bar Chart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Final Performance Comparison (Last Epoch)', fontsize=16, fontweight='bold')

# Get final epoch data
final_carot = df_carot.iloc[-1]
final_ldreg = df_ldreg.iloc[-1]

# Accuracy comparison
acc_metrics = ['ImageNet Accuracy', 'ImageNetV2 Accuracy', 'ImageNetR Accuracy', 
               'ImageNetA Accuracy', 'ImageNetSketch Accuracy']
carot_accs = [final_carot[metric] for metric in acc_metrics]
ldreg_accs = [final_ldreg[metric] for metric in acc_metrics]

x = np.arange(len(acc_metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, carot_accs, width, label='CaRot', color=colors['CaRot'], alpha=0.8)
bars2 = ax1.bar(x + width/2, ldreg_accs, width, label='LDReg', color=colors['LDReg'], alpha=0.8)

ax1.set_title('Final Accuracy Comparison', fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(' Accuracy', '') for m in acc_metrics], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# ECE comparison
ece_metrics = ['ImageNet ECE', 'ImageNetV2 ECE', 'ImageNetR ECE', 
               'ImageNetA ECE', 'ImageNetSketch ECE']
carot_eces = [final_carot[metric] for metric in ece_metrics]
ldreg_eces = [final_ldreg[metric] for metric in ece_metrics]

bars3 = ax2.bar(x - width/2, carot_eces, width, label='CaRot', color=colors['CaRot'], alpha=0.8)
bars4 = ax2.bar(x + width/2, ldreg_eces, width, label='LDReg', color=colors['LDReg'], alpha=0.8)

ax2.set_title('Final ECE Comparison (Lower is Better)', fontweight='bold')
ax2.set_ylabel('Expected Calibration Error')
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ECE', '') for m in ece_metrics], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_plots/05_final_performance_bars.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Learning Rate Schedule Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(df_carot['epoch'], df_carot['Final LR'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax.semilogy(df_ldreg['epoch'], df_ldreg['Final LR'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax.set_title('Learning Rate Schedule Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate (log scale)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_plots/06_learning_rate_schedule.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 7: LDReg-specific Metrics (only available for LDReg)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('LDReg-specific Metrics', fontsize=16, fontweight='bold')

# LDReg Loss
ax1.plot(df_ldreg['epoch'], df_ldreg['Avg LDReg Loss'], 's-', color=colors['LDReg'], alpha=0.8)
ax1.set_title('LDReg Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('LDReg Loss')
ax1.grid(True, alpha=0.3)

# Mean LID (Local Intrinsic Dimensionality)
ax2.plot(df_ldreg['epoch'], df_ldreg['Avg Mean LID'], 's-', color=colors['LDReg'], alpha=0.8)
ax2.set_title('Mean Local Intrinsic Dimensionality (LID)', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean LID')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/07_ldreg_specific_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 8: Accuracy vs ECE Scatter Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Accuracy vs ECE Trade-off Analysis', fontsize=16, fontweight='bold')

datasets_full = ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']
positions_scatter = [(0,0), (0,1), (0,2), (1,0), (1,1)]

for i, (dataset, pos) in enumerate(zip(datasets_full, positions_scatter)):
    ax = axes[pos]
    acc_col = f'{dataset} Accuracy'
    ece_col = f'{dataset} ECE'
    
    # Plot trajectory for each method
    ax.plot(df_carot[ece_col], df_carot[acc_col], 'o-', color=colors['CaRot'], 
            label='CaRot', alpha=0.7, markersize=6)
    ax.plot(df_ldreg[ece_col], df_ldreg[acc_col], 's-', color=colors['LDReg'], 
            label='LDReg', alpha=0.7, markersize=6)
    
    # Mark start and end points
    ax.scatter(df_carot[ece_col].iloc[0], df_carot[acc_col].iloc[0], 
              color=colors['CaRot'], s=100, marker='o', edgecolor='black', linewidth=2, alpha=0.9)
    ax.scatter(df_carot[ece_col].iloc[-1], df_carot[acc_col].iloc[-1], 
              color=colors['CaRot'], s=100, marker='*', edgecolor='black', linewidth=2, alpha=0.9)
    ax.scatter(df_ldreg[ece_col].iloc[0], df_ldreg[acc_col].iloc[0], 
              color=colors['LDReg'], s=100, marker='s', edgecolor='black', linewidth=2, alpha=0.9)
    ax.scatter(df_ldreg[ece_col].iloc[-1], df_ldreg[acc_col].iloc[-1], 
              color=colors['LDReg'], s=100, marker='*', edgecolor='black', linewidth=2, alpha=0.9)
    
    ax.set_title(f'{dataset}', fontweight='bold')
    ax.set_xlabel('ECE (Lower is Better)')
    ax.set_ylabel('Accuracy (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove the empty subplot
axes[1,2].remove()

plt.tight_layout()
plt.savefig('comparison_plots/08_accuracy_vs_ece_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 9: Convergence Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Convergence Analysis: Rate of Improvement', fontsize=16, fontweight='bold')

# Calculate improvement rates (difference between consecutive epochs)
carot_acc_diff = df_carot['ImageNet Accuracy'].diff()
ldreg_acc_diff = df_ldreg['ImageNet Accuracy'].diff()
carot_loss_diff = -df_carot['Avg Total Loss'].diff()  # Negative because we want loss reduction
ldreg_loss_diff = -df_ldreg['Avg Total Loss'].diff()

# Accuracy improvement
axes[0,0].plot(df_carot['epoch'][1:], carot_acc_diff[1:], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[0,0].plot(df_ldreg['epoch'][1:], ldreg_acc_diff[1:], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[0,0].set_title('ImageNet Accuracy Improvement per Epoch', fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy Improvement')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Loss reduction
axes[0,1].plot(df_carot['epoch'][1:], carot_loss_diff[1:], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[0,1].plot(df_ldreg['epoch'][1:], ldreg_loss_diff[1:], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[0,1].set_title('Total Loss Reduction per Epoch', fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss Reduction')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Cumulative accuracy improvement
axes[1,0].plot(df_carot['epoch'], df_carot['ImageNet Accuracy'] - df_carot['ImageNet Accuracy'].iloc[0], 
               'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[1,0].plot(df_ldreg['epoch'], df_ldreg['ImageNet Accuracy'] - df_ldreg['ImageNet Accuracy'].iloc[0], 
               's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[1,0].set_title('Cumulative Accuracy Improvement', fontweight='bold')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Cumulative Accuracy Gain')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Relative loss reduction
carot_rel_loss = (df_carot['Avg Total Loss'].iloc[0] - df_carot['Avg Total Loss']) / df_carot['Avg Total Loss'].iloc[0]
ldreg_rel_loss = (df_ldreg['Avg Total Loss'].iloc[0] - df_ldreg['Avg Total Loss']) / df_ldreg['Avg Total Loss'].iloc[0]

axes[1,1].plot(df_carot['epoch'], carot_rel_loss, 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
axes[1,1].plot(df_ldreg['epoch'], ldreg_rel_loss, 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
axes[1,1].set_title('Relative Loss Reduction', fontweight='bold')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Relative Loss Reduction')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plots/09_convergence_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 10: Comprehensive Summary Dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('CaRot vs LDReg: Comprehensive Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)

# 1. Training Loss Evolution
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_carot['epoch'], df_carot['Avg Total Loss'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax1.plot(df_ldreg['epoch'], df_ldreg['Avg Total Loss'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax1.set_title('Training Loss', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. ImageNet Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_carot['epoch'], df_carot['ImageNet Accuracy'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax2.plot(df_ldreg['epoch'], df_ldreg['ImageNet Accuracy'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax2.set_title('ImageNet Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Calibration Quality
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(df_carot['epoch'], df_carot['ImageNet ECE'], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
ax3.plot(df_ldreg['epoch'], df_ldreg['ImageNet ECE'], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
ax3.set_title('Calibration (ECE)', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('ECE')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Robustness Summary (final epoch)
ax4 = fig.add_subplot(gs[0, 3])
robust_datasets = ['ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']
carot_robust = [final_carot[f'{d} Accuracy'] for d in robust_datasets]
ldreg_robust = [final_ldreg[f'{d} Accuracy'] for d in robust_datasets]

x_robust = np.arange(len(robust_datasets))
width = 0.35
ax4.bar(x_robust - width/2, carot_robust, width, label='CaRot', color=colors['CaRot'], alpha=0.8)
ax4.bar(x_robust + width/2, ldreg_robust, width, label='LDReg', color=colors['LDReg'], alpha=0.8)
ax4.set_title('Robustness (Final)', fontweight='bold')
ax4.set_ylabel('Accuracy')
ax4.set_xticks(x_robust)
ax4.set_xticklabels([d.replace('ImageNet', '') for d in robust_datasets], rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5-8. Individual robustness datasets
robust_positions = [gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
for i, (dataset, pos) in enumerate(zip(robust_datasets, robust_positions)):
    ax = fig.add_subplot(pos)
    acc_col = f'{dataset} Accuracy'
    ax.plot(df_carot['epoch'], df_carot[acc_col], 'o-', color=colors['CaRot'], label='CaRot', alpha=0.8)
    ax.plot(df_ldreg['epoch'], df_ldreg[acc_col], 's-', color=colors['LDReg'], label='LDReg', alpha=0.8)
    ax.set_title(f'{dataset.replace("ImageNet", "")} Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    if i == 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

# 9. Performance Summary Table
ax9 = fig.add_subplot(gs[2, :2])
ax9.axis('tight')
ax9.axis('off')

# Create summary table
metrics = ['ImageNet Acc', 'ImageNet ECE', 'ImageNetV2 Acc', 'ImageNetR Acc', 
           'ImageNetA Acc', 'ImageNetSketch Acc', 'Final Loss']
carot_values = [
    f"{final_carot['ImageNet Accuracy']:.4f}",
    f"{final_carot['ImageNet ECE']:.4f}",
    f"{final_carot['ImageNetV2 Accuracy']:.4f}",
    f"{final_carot['ImageNetR Accuracy']:.4f}",
    f"{final_carot['ImageNetA Accuracy']:.4f}",
    f"{final_carot['ImageNetSketch Accuracy']:.4f}",
    f"{final_carot['Avg Total Loss']:.4f}"
]
ldreg_values = [
    f"{final_ldreg['ImageNet Accuracy']:.4f}",
    f"{final_ldreg['ImageNet ECE']:.4f}",
    f"{final_ldreg['ImageNetV2 Accuracy']:.4f}",
    f"{final_ldreg['ImageNetR Accuracy']:.4f}",
    f"{final_ldreg['ImageNetA Accuracy']:.4f}",
    f"{final_ldreg['ImageNetSketch Accuracy']:.4f}",
    f"{final_ldreg['Avg Total Loss']:.4f}"
]

table_data = [['Metric', 'CaRot', 'LDReg']]
for i, metric in enumerate(metrics):
    table_data.append([metric, carot_values[i], ldreg_values[i]])

table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the header row
for i in range(3):
    table[(0, i)].set_facecolor('#E8E8E8')
    table[(0, i)].set_text_props(weight='bold')

ax9.set_title('Final Performance Summary', fontweight='bold', pad=20)

# 10. Method Comparison Radar Chart
ax10 = fig.add_subplot(gs[2, 2:], projection='polar')

# Normalize metrics for radar chart (higher is better for all)
radar_metrics = ['ImageNet Acc', 'Robustness', 'Calibration', 'Convergence']
carot_radar = [
    final_carot['ImageNet Accuracy'],
    np.mean([final_carot[f'{d} Accuracy'] for d in robust_datasets]),
    1 - final_carot['ImageNet ECE'],  # Invert ECE so higher is better
    (df_carot['ImageNet Accuracy'].iloc[-1] - df_carot['ImageNet Accuracy'].iloc[0]) / len(df_carot)
]
ldreg_radar = [
    final_ldreg['ImageNet Accuracy'],
    np.mean([final_ldreg[f'{d} Accuracy'] for d in robust_datasets]),
    1 - final_ldreg['ImageNet ECE'],  # Invert ECE so higher is better
    (df_ldreg['ImageNet Accuracy'].iloc[-1] - df_ldreg['ImageNet Accuracy'].iloc[0]) / len(df_ldreg)
]

# Normalize to 0-1 scale
all_values = carot_radar + ldreg_radar
min_val, max_val = min(all_values), max(all_values)
carot_radar_norm = [(v - min_val) / (max_val - min_val) for v in carot_radar]
ldreg_radar_norm = [(v - min_val) / (max_val - min_val) for v in ldreg_radar]

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
carot_radar_norm += carot_radar_norm[:1]  # Complete the circle
ldreg_radar_norm += ldreg_radar_norm[:1]
angles += angles[:1]

ax10.plot(angles, carot_radar_norm, 'o-', linewidth=2, label='CaRot', color=colors['CaRot'])
ax10.fill(angles, carot_radar_norm, alpha=0.25, color=colors['CaRot'])
ax10.plot(angles, ldreg_radar_norm, 's-', linewidth=2, label='LDReg', color=colors['LDReg'])
ax10.fill(angles, ldreg_radar_norm, alpha=0.25, color=colors['LDReg'])

ax10.set_xticks(angles[:-1])
ax10.set_xticklabels(radar_metrics)
ax10.set_ylim(0, 1)
ax10.set_title('Overall Performance Comparison', fontweight='bold', pad=20)
ax10.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax10.grid(True)

plt.savefig('comparison_plots/10_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 11: Radar Chart - Accuracy Across All Datasets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
fig.suptitle('Accuracy Comparison Across All Datasets', fontsize=16, fontweight='bold')

# Define all accuracy metrics
accuracy_datasets = ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']
accuracy_labels = ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']

# Get final epoch accuracies
final_carot = df_carot.iloc[-1]
final_ldreg = df_ldreg.iloc[-1]

carot_accuracies = [final_carot[f'{dataset} Accuracy'] for dataset in accuracy_datasets]
ldreg_accuracies = [final_ldreg[f'{dataset} Accuracy'] for dataset in accuracy_datasets]

# Set up angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(accuracy_datasets), endpoint=False).tolist()

# Complete the circle by adding the first value at the end
carot_accuracies_circle = carot_accuracies + [carot_accuracies[0]]
ldreg_accuracies_circle = ldreg_accuracies + [ldreg_accuracies[0]]
angles_circle = angles + [angles[0]]

# Plot 1: Final Epoch Comparison
ax1.plot(angles_circle, carot_accuracies_circle, 'o-', linewidth=3, 
         label='CaRot', color=colors['CaRot'], markersize=8)
ax1.fill(angles_circle, carot_accuracies_circle, alpha=0.25, color=colors['CaRot'])

ax1.plot(angles_circle, ldreg_accuracies_circle, 's-', linewidth=3, 
         label='LDReg', color=colors['LDReg'], markersize=8)
ax1.fill(angles_circle, ldreg_accuracies_circle, alpha=0.25, color=colors['LDReg'])

# Customize the radar chart
ax1.set_xticks(angles)
ax1.set_xticklabels(accuracy_labels, fontsize=11)
ax1.set_ylim(0, 1)
ax1.set_title('Final Epoch Accuracy', fontweight='bold', pad=20, fontsize=14)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax1.grid(True, alpha=0.3)

# Add value labels on the radar chart
for angle, carot_acc, ldreg_acc in zip(angles, carot_accuracies, ldreg_accuracies):
    # CaRot values
    ax1.text(angle, carot_acc + 0.02, f'{carot_acc:.3f}', 
             ha='center', va='center', fontsize=9, color=colors['CaRot'], fontweight='bold')
    # LDReg values
    ax1.text(angle, ldreg_acc - 0.03, f'{ldreg_acc:.3f}', 
             ha='center', va='center', fontsize=9, color=colors['LDReg'], fontweight='bold')

# Set radial ticks
ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)

# Plot 2: Average Accuracy Across Training
# Calculate mean accuracy across all epochs for each dataset
carot_mean_accuracies = [df_carot[f'{dataset} Accuracy'].mean() for dataset in accuracy_datasets]
ldreg_mean_accuracies = [df_ldreg[f'{dataset} Accuracy'].mean() for dataset in accuracy_datasets]

carot_mean_circle = carot_mean_accuracies + [carot_mean_accuracies[0]]
ldreg_mean_circle = ldreg_mean_accuracies + [ldreg_mean_accuracies[0]]

ax2.plot(angles_circle, carot_mean_circle, 'o-', linewidth=3, 
         label='CaRot', color=colors['CaRot'], markersize=8)
ax2.fill(angles_circle, carot_mean_circle, alpha=0.25, color=colors['CaRot'])

ax2.plot(angles_circle, ldreg_mean_circle, 's-', linewidth=3, 
         label='LDReg', color=colors['LDReg'], markersize=8)
ax2.fill(angles_circle, ldreg_mean_circle, alpha=0.25, color=colors['LDReg'])

ax2.set_xticks(angles)
ax2.set_xticklabels(accuracy_labels, fontsize=11)
ax2.set_ylim(0, 1)
ax2.set_title('Average Accuracy Across Training', fontweight='bold', pad=20, fontsize=14)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax2.grid(True, alpha=0.3)

# Add value labels
for angle, carot_acc, ldreg_acc in zip(angles, carot_mean_accuracies, ldreg_mean_accuracies):
    ax2.text(angle, carot_acc + 0.02, f'{carot_acc:.3f}', 
             ha='center', va='center', fontsize=9, color=colors['CaRot'], fontweight='bold')
    ax2.text(angle, ldreg_acc - 0.03, f'{ldreg_acc:.3f}', 
             ha='center', va='center', fontsize=9, color=colors['LDReg'], fontweight='bold')

ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)

plt.tight_layout()
plt.savefig('comparison_plots/11_accuracy_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 12: Enhanced Radar Chart with Improvement Analysis
fig, ax = plt.subplots(1, 1, figsize=(16, 14), subplot_kw=dict(projection='polar'))
fig.suptitle('Comprehensive Accuracy Analysis: Final vs Initial Performance', fontsize=16, fontweight='bold')

# Get initial epoch accuracies
initial_carot = df_carot.iloc[0]
initial_ldreg = df_ldreg.iloc[0]

initial_carot_accuracies = [initial_carot[f'{dataset} Accuracy'] for dataset in accuracy_datasets]
initial_ldreg_accuracies = [initial_ldreg[f'{dataset} Accuracy'] for dataset in accuracy_datasets]

# Complete circles
initial_carot_circle = initial_carot_accuracies + [initial_carot_accuracies[0]]
initial_ldreg_circle = initial_ldreg_accuracies + [initial_ldreg_accuracies[0]]

# Plot initial performance (dashed lines)
ax.plot(angles_circle, initial_carot_circle, '--', linewidth=2, 
        label='CaRot (Initial)', color=colors['CaRot'], alpha=0.6)
ax.plot(angles_circle, initial_ldreg_circle, '--', linewidth=2, 
        label='LDReg (Initial)', color=colors['LDReg'], alpha=0.6)

# Plot final performance (solid lines)
ax.plot(angles_circle, carot_accuracies_circle, 'o-', linewidth=3, 
        label='CaRot (Final)', color=colors['CaRot'], markersize=8)
ax.fill(angles_circle, carot_accuracies_circle, alpha=0.15, color=colors['CaRot'])

ax.plot(angles_circle, ldreg_accuracies_circle, 's-', linewidth=3, 
        label='LDReg (Final)', color=colors['LDReg'], markersize=8)
ax.fill(angles_circle, ldreg_accuracies_circle, alpha=0.15, color=colors['LDReg'])

# Customize
ax.set_xticks(angles)
ax.set_xticklabels(accuracy_labels, fontsize=12)
ax.set_ylim(0, 1.2)  # Extended range to accommodate text boxes
ax.set_title('Initial vs Final Accuracy Comparison', fontweight='bold', pad=30, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

# Position text boxes in outer ring to avoid overlap
# Use alternating inner and outer positions
outer_radius = 1.05
inner_radius = 0.85

for i, (angle, dataset) in enumerate(zip(angles, accuracy_datasets)):
    carot_improvement = carot_accuracies[i] - initial_carot_accuracies[i]
    ldreg_improvement = ldreg_accuracies[i] - initial_ldreg_accuracies[i]
    
    # Alternate between inner and outer positions
    if i % 2 == 0:
        text_radius = outer_radius
    else:
        text_radius = inner_radius
    
    # Create improvement text
    improvement_text = f'CaRot: +{carot_improvement:.3f}\nLDReg: +{ldreg_improvement:.3f}'
    
    # Position text box
    ax.text(angle, text_radius, improvement_text, 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.95, 
                     edgecolor='gray', linewidth=1))
    
    # Draw connecting line from text to data point
    data_point_radius = max(carot_accuracies[i], ldreg_accuracies[i]) + 0.02
    ax.plot([angle, angle], [data_point_radius, text_radius - 0.05], 
            color='gray', linestyle=':', alpha=0.6, linewidth=1)

# Remove the center text box to avoid clutter
# Instead, add a legend box outside the plot
fig.text(0.02, 0.5, 
         f"Average Improvement:\nCaRot: +{np.mean([carot_accuracies[i] - initial_carot_accuracies[i] for i in range(len(accuracy_datasets))]):.3f}\nLDReg: +{np.mean([ldreg_accuracies[i] - initial_ldreg_accuracies[i] for i in range(len(accuracy_datasets))]):.3f}", 
         fontsize=12, fontweight='bold', va='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8, 
                  edgecolor='navy', linewidth=2))

ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

plt.tight_layout()
plt.savefig('comparison_plots/12_accuracy_radar_improvement.png', dpi=150, bbox_inches='tight')
plt.close()

# Alternative version: Separate improvement table
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                               gridspec_kw={'width_ratios': [2, 1]},
                               subplot_kw={'projection': 'polar'})
fig.suptitle('Accuracy Analysis: Performance and Improvements', fontsize=16, fontweight='bold')

# Main radar chart (cleaner without text boxes)
ax1.plot(angles_circle, initial_carot_circle, '--', linewidth=2, 
         label='CaRot (Initial)', color=colors['CaRot'], alpha=0.6)
ax1.plot(angles_circle, initial_ldreg_circle, '--', linewidth=2, 
         label='LDReg (Initial)', color=colors['LDReg'], alpha=0.6)

ax1.plot(angles_circle, carot_accuracies_circle, 'o-', linewidth=3, 
         label='CaRot (Final)', color=colors['CaRot'], markersize=8)
ax1.fill(angles_circle, carot_accuracies_circle, alpha=0.15, color=colors['CaRot'])

ax1.plot(angles_circle, ldreg_accuracies_circle, 's-', linewidth=3, 
         label='LDReg (Final)', color=colors['LDReg'], markersize=8)
ax1.fill(angles_circle, ldreg_accuracies_circle, alpha=0.15, color=colors['LDReg'])

ax1.set_xticks(angles)
ax1.set_xticklabels(accuracy_labels, fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_title('Accuracy Comparison', fontweight='bold', pad=20, fontsize=14)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)

# Improvement table on the right
ax2.remove()  # Remove the polar subplot
ax2 = fig.add_subplot(1, 2, 2)  # Add regular subplot
ax2.axis('off')

# Create improvement table
improvement_data = []
improvement_data.append(['Dataset', 'CaRot Δ', 'LDReg Δ'])
for i, dataset in enumerate(accuracy_datasets):
    carot_imp = carot_accuracies[i] - initial_carot_accuracies[i]
    ldreg_imp = ldreg_accuracies[i] - initial_ldreg_accuracies[i]
    dataset_short = dataset.replace('ImageNet', '').replace('Net', '') or 'ImageNet'
    improvement_data.append([dataset_short, f'+{carot_imp:.3f}', f'+{ldreg_imp:.3f}'])

# Add average row
avg_carot = np.mean([carot_accuracies[i] - initial_carot_accuracies[i] for i in range(len(accuracy_datasets))])
avg_ldreg = np.mean([ldreg_accuracies[i] - initial_ldreg_accuracies[i] for i in range(len(accuracy_datasets))])
improvement_data.append(['Average', f'+{avg_carot:.3f}', f'+{avg_ldreg:.3f}'])

table = ax2.table(cellText=improvement_data, cellLoc='center', loc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Style the table
for i in range(3):
    table[(0, i)].set_facecolor('#E8E8E8')
    table[(0, i)].set_text_props(weight='bold')

# Highlight the average row
for i in range(3):
    table[(len(improvement_data)-1, i)].set_facecolor('#D4EDDA')
    table[(len(improvement_data)-1, i)].set_text_props(weight='bold')

ax2.set_title('Accuracy Improvements\n(Final - Initial)', fontweight='bold', pad=20, fontsize=14)

plt.tight_layout()
plt.savefig('comparison_plots/12_accuracy_radar_improvement_v2.png', dpi=150, bbox_inches='tight')
plt.close()

print("All 10 scientific plots have been generated and saved in the 'comparison_plots' directory!")
print("\nPlot descriptions:")
print("1. Loss Components - Training loss breakdown comparison")
print("2. ImageNet Performance - Main accuracy and calibration metrics")
print("3. Robustness Accuracy - Performance across ImageNet variants")
print("4. Calibration Quality - ECE across all datasets")
print("5. Final Performance Bars - Side-by-side comparison of final metrics")
print("6. Learning Rate Schedule - LR decay comparison")
print("7. LDReg Specific Metrics - Unique LDReg measurements")
print("8. Accuracy vs ECE Scatter - Trade-off analysis")
print("9. Convergence Analysis - Training dynamics and improvement rates")
print("10. Comprehensive Dashboard - Complete overview with summary table and radar chart")
print("11. Accuracy Radar Chart - Final epoch and average accuracy comparison")
print("12. Accuracy Radar with Improvement - Shows initial vs final performance with improvement metrics")