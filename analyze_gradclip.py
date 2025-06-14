import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create the data
data = """Name,ImageNet Accuracy,ImageNetV2 Accuracy,ImageNetSketch Accuracy,ImageNetA Accuracy,ImageNetR Accuracy
carot_BMA_gradCLIP_0.0001_withText,0.803,0.7208,0.5343,0.5464,0.7994
carot_EMA_gradCLIP_0.001_withText,0.7989,0.716,0.5269,0.5349,0.7955
carot_EMA_gradCLIP_0.0005_withText,0.7976,0.7155,0.5274,0.5383,0.797
carot_EMA_gradCLIP_0.0001_withText,0.7904,0.7104,0.5266,0.5439,0.7966
carot_EMA_gradCLIP_0.0001_withoutText,0.7859,0.7072,0.5254,0.5413,0.797
carot_textFreeze,0.8258,0.7405,0.5266,0.5091,0.7756
carot_base,0.8302,0.7412,0.5295,0.5164,0.7786
flyp_ldreg_gradCLIP_0.0001_withText,0.8186,0.7294,0.5034,0.5204,0.7367
flyp_ldreg_gradCLIP_0.00001_withText,0.8065,0.7185,0.5076,0.5331,0.7599
flyp_gradCLIP_0.00001_withText,0.8084,0.7224,0.5056,0.5315,0.7616
flyp_gradCLIP_0.0001_withText,0.8196,0.7298,0.5018,0.5205,0.7306
flyp_base_withText,0.8256,0.7294,0.4946,0.4804,0.7149"""

# Read data into DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Calculate average OOD accuracy
df['Avg OOD Accuracy'] = df[['ImageNetV2 Accuracy', 'ImageNetSketch Accuracy', 
                              'ImageNetA Accuracy', 'ImageNetR Accuracy']].mean(axis=1)

# Extract method type and configuration details
df['Method'] = df['Name'].apply(lambda x: x.split('_')[0])
df['Has_GradCLIP'] = df['Name'].str.contains('gradCLIP')

# Extract max gradient norm value
def extract_max_grad_norm(name):
    if 'gradCLIP' not in name:
        return None
    parts = name.split('_')
    for i, part in enumerate(parts):
        if part == 'gradCLIP' and i + 1 < len(parts):
            try:
                return float(parts[i + 1])
            except ValueError:
                return None
    return None

df['Max_Grad_Norm'] = df['Name'].apply(extract_max_grad_norm)
df['Configuration'] = df['Name'].apply(lambda x: '_'.join(x.split('_')[1:]))

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Comparison of ImageNet vs Average OOD Accuracy
ax1 = plt.subplot(3, 3, 1)
colors = ['#FF6B6B' if 'carot' in name else '#4ECDC4' for name in df['Name']]
scatter = ax1.scatter(df['ImageNet Accuracy'], df['Avg OOD Accuracy'], 
                     c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1)

# Add labels for specific points
for idx, row in df.iterrows():
    if 'base' in row['Name'] or 'gradCLIP_0.0001' in row['Name']:
        ax1.annotate(row['Name'].split('_')[1] if 'base' not in row['Name'] else 'base', 
                    (row['ImageNet Accuracy'], row['Avg OOD Accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('ImageNet Accuracy', fontsize=12)
ax1.set_ylabel('Average OOD Accuracy', fontsize=12)
ax1.set_title('ImageNet vs OOD Performance Trade-off', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add legend
carot_patch = mpatches.Patch(color='#FF6B6B', label='CAROT')
flyp_patch = mpatches.Patch(color='#4ECDC4', label='FLYP')
ax1.legend(handles=[carot_patch, flyp_patch])

# 2. Max Gradient Norm Effect on Performance
ax2 = plt.subplot(3, 3, 2)
grad_clip_df = df[df['Has_GradCLIP']].copy()

for method in ['carot', 'flyp']:
    method_df = grad_clip_df[grad_clip_df['Method'] == method]
    if len(method_df) > 0:
        # Group by max grad norm and take mean if multiple configs per norm
        norm_grouped = method_df.groupby('Max_Grad_Norm')[['ImageNet Accuracy', 'Avg OOD Accuracy']].mean()
        
        ax2.plot(norm_grouped.index, norm_grouped['ImageNet Accuracy'], 
                marker='o', label=f'{method.upper()} - ImageNet', linewidth=2, markersize=8)
        ax2.plot(norm_grouped.index, norm_grouped['Avg OOD Accuracy'], 
                marker='s', label=f'{method.upper()} - Avg OOD', linewidth=2, markersize=8, linestyle='--')

ax2.set_xlabel('Max Gradient Norm', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Effect of Gradient Clipping Threshold', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Comparison of all OOD datasets
ax3 = plt.subplot(3, 3, 3)
ood_cols = ['ImageNetV2 Accuracy', 'ImageNetSketch Accuracy', 'ImageNetA Accuracy', 'ImageNetR Accuracy']
selected_models = ['carot_base', 'carot_EMA_gradCLIP_0.0001_withText', 
                   'flyp_base_withText', 'flyp_gradCLIP_0.0001_withText']
selected_df = df[df['Name'].isin(selected_models)]

x = np.arange(len(ood_cols))
width = 0.2

for i, model in enumerate(selected_models):
    model_data = selected_df[selected_df['Name'] == model][ood_cols].values[0]
    ax3.bar(x + i*width, model_data, width, label=model.replace('_', ' '))

ax3.set_xlabel('Dataset', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Performance Across OOD Datasets', fontsize=14, fontweight='bold')
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(['V2', 'Sketch', 'A', 'R'])
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Heatmap of all accuracies
ax4 = plt.subplot(3, 3, 4)
heatmap_data = df.set_index('Name')[['ImageNet Accuracy', 'ImageNetV2 Accuracy', 
                                      'ImageNetSketch Accuracy', 'ImageNetA Accuracy', 
                                      'ImageNetR Accuracy', 'Avg OOD Accuracy']]
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4, cbar_kws={'label': 'Accuracy'})
ax4.set_title('Performance Heatmap Across All Datasets', fontsize=14, fontweight='bold')
ax4.set_xlabel('')

# 5. Gradient Clipping vs No Gradient Clipping
ax5 = plt.subplot(3, 3, 5)
metrics = ['ImageNet Accuracy', 'Avg OOD Accuracy']
carot_base_vals = df[df['Name'] == 'carot_base'][metrics].values[0]
carot_gradclip_vals = df[df['Name'] == 'carot_EMA_gradCLIP_0.0001_withText'][metrics].values[0]
flyp_base_vals = df[df['Name'] == 'flyp_base_withText'][metrics].values[0]
flyp_gradclip_vals = df[df['Name'] == 'flyp_gradCLIP_0.0001_withText'][metrics].values[0]

x = np.arange(len(metrics))
width = 0.35

ax5.bar(x - width/2, [carot_base_vals[0], carot_base_vals[1]], width, label='CAROT Base', color='#FF6B6B')
ax5.bar(x + width/2, [carot_gradclip_vals[0], carot_gradclip_vals[1]], width, label='CAROT GradCLIP', color='#FF6B6B', alpha=0.6)
ax5.bar(x - width/2 + 2, [flyp_base_vals[0], flyp_base_vals[1]], width, label='FLYP Base', color='#4ECDC4')
ax5.bar(x + width/2 + 2, [flyp_gradclip_vals[0], flyp_gradclip_vals[1]], width, label='FLYP GradCLIP', color='#4ECDC4', alpha=0.6)

ax5.set_ylabel('Accuracy', fontsize=12)
ax5.set_title('Base vs Gradient Clipping Performance', fontsize=14, fontweight='bold')
ax5.set_xticks([0, 1, 2, 3])
ax5.set_xticklabels(['CAROT\nImageNet', 'CAROT\nAvg OOD', 'FLYP\nImageNet', 'FLYP\nAvg OOD'])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Box plot for OOD performance distribution
ax6 = plt.subplot(3, 3, 6)
ood_data = []
labels = []
for _, row in df.iterrows():
    ood_values = [row['ImageNetV2 Accuracy'], row['ImageNetSketch Accuracy'], 
                  row['ImageNetA Accuracy'], row['ImageNetR Accuracy']]
    ood_data.append(ood_values)
    labels.append(row['Name'].split('_')[1] if len(row['Name'].split('_')) > 1 else 'base')

positions = list(range(len(ood_data)))
bp = ax6.boxplot(ood_data, positions=positions, patch_artist=True)

# Color boxes
for i, (box, label) in enumerate(zip(bp['boxes'], df['Name'])):
    if 'carot' in label:
        box.set_facecolor('#FF6B6B')
    else:
        box.set_facecolor('#4ECDC4')
    box.set_alpha(0.7)

ax6.set_xticklabels(labels, rotation=45, ha='right')
ax6.set_ylabel('OOD Accuracy', fontsize=12)
ax6.set_title('OOD Performance Distribution by Method', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Performance improvement with gradient clipping
ax7 = plt.subplot(3, 3, 7)
improvements = []
categories = []

# Calculate improvements for CAROT
carot_base_ood = df[df['Name'] == 'carot_base']['Avg OOD Accuracy'].values[0]
carot_gradclip_ood = df[df['Name'] == 'carot_EMA_gradCLIP_0.0001_withText']['Avg OOD Accuracy'].values[0]
improvements.append((carot_gradclip_ood - carot_base_ood) / carot_base_ood * 100)
categories.append('CAROT OOD')

carot_base_in = df[df['Name'] == 'carot_base']['ImageNet Accuracy'].values[0]
carot_gradclip_in = df[df['Name'] == 'carot_EMA_gradCLIP_0.0001_withText']['ImageNet Accuracy'].values[0]
improvements.append((carot_gradclip_in - carot_base_in) / carot_base_in * 100)
categories.append('CAROT IN')

# Calculate improvements for FLYP
flyp_base_ood = df[df['Name'] == 'flyp_base_withText']['Avg OOD Accuracy'].values[0]
flyp_gradclip_ood = df[df['Name'] == 'flyp_gradCLIP_0.0001_withText']['Avg OOD Accuracy'].values[0]
improvements.append((flyp_gradclip_ood - flyp_base_ood) / flyp_base_ood * 100)
categories.append('FLYP OOD')

flyp_base_in = df[df['Name'] == 'flyp_base_withText']['ImageNet Accuracy'].values[0]
flyp_gradclip_in = df[df['Name'] == 'flyp_gradCLIP_0.0001_withText']['ImageNet Accuracy'].values[0]
improvements.append((flyp_gradclip_in - flyp_base_in) / flyp_base_in * 100)
categories.append('FLYP IN')

colors_imp = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
bars = ax7.bar(categories, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1)

# Add value labels on bars
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:.2f}%', ha='center', va='bottom' if height > 0 else 'top')

ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax7.set_ylabel('Improvement (%)', fontsize=12)
ax7.set_title('Performance Change with Gradient Clipping', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Radar chart for selected models
ax8 = plt.subplot(3, 3, 8, projection='polar')
categories_radar = ['ImageNet', 'V2', 'Sketch', 'A', 'R']
models_to_compare = ['carot_base', 'carot_EMA_gradCLIP_0.0001_withText', 
                     'flyp_base_withText', 'flyp_gradCLIP_0.0001_withText']
colors_radar = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
linestyles = ['-', '--', '-', '--']

angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
angles += angles[:1]

for model, color, ls in zip(models_to_compare, colors_radar, linestyles):
    values = df[df['Name'] == model][['ImageNet Accuracy', 'ImageNetV2 Accuracy', 
                                      'ImageNetSketch Accuracy', 'ImageNetA Accuracy', 
                                      'ImageNetR Accuracy']].values[0].tolist()
    values += values[:1]
    ax8.plot(angles, values, color=color, linewidth=2, linestyle=ls, 
            label=model.replace('_', ' '))
    ax8.fill(angles, values, color=color, alpha=0.1)

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(categories_radar)
ax8.set_ylim(0, 1)
ax8.set_title('Performance Profile Comparison', fontsize=14, fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax8.grid(True)

# 9. Summary statistics table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('tight')
ax9.axis('off')

summary_data = []
for method in ['carot', 'flyp']:
    base_model = f'{method}_base' if method == 'carot' else f'{method}_base_withText'
    base_row = df[df['Name'] == base_model].iloc[0]
    
    gradclip_models = df[(df['Method'] == method) & (df['Has_GradCLIP'])]
    if len(gradclip_models) > 0:
        best_gradclip = gradclip_models.loc[gradclip_models['Avg OOD Accuracy'].idxmax()]
        
        summary_data.append([
            method.upper(),
            f"{base_row['ImageNet Accuracy']:.3f}",
            f"{base_row['Avg OOD Accuracy']:.3f}",
            f"{best_gradclip['ImageNet Accuracy']:.3f}",
            f"{best_gradclip['Avg OOD Accuracy']:.3f}",
            f"{((best_gradclip['Avg OOD Accuracy'] - base_row['Avg OOD Accuracy']) / base_row['Avg OOD Accuracy'] * 100):.1f}%"
        ])

table = ax9.table(cellText=summary_data,
                  colLabels=['Method', 'Base IN', 'Base OOD', 'Best GradCLIP IN', 'Best GradCLIP OOD', 'OOD Improve'],
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax9.set_title('Summary: Base vs Best Gradient Clipping', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('gradient_clipping_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the dataframe with average OOD accuracy
print("\nDataFrame with Average OOD Accuracy:")
print(df[['Name', 'ImageNet Accuracy', 'Avg OOD Accuracy']].to_string(index=False))

# Print key insights
print("\n\nKey Insights:")
print("="*50)
print("\n1. CAROT Method:")
print(f"   - Base model OOD accuracy: {df[df['Name'] == 'carot_base']['Avg OOD Accuracy'].values[0]:.4f}")
print(f"   - Best gradient clipping OOD: {df[(df['Method'] == 'carot') & (df['Has_GradCLIP'])]['Avg OOD Accuracy'].max():.4f}")
print(f"   - Improvement: {((df[(df['Method'] == 'carot') & (df['Has_GradCLIP'])]['Avg OOD Accuracy'].max() - df[df['Name'] == 'carot_base']['Avg OOD Accuracy'].values[0]) / df[df['Name'] == 'carot_base']['Avg OOD Accuracy'].values[0] * 100):.2f}%")

print("\n2. FLYP Method:")
print(f"   - Base model OOD accuracy: {df[df['Name'] == 'flyp_base_withText']['Avg OOD Accuracy'].values[0]:.4f}")
print(f"   - Best gradient clipping OOD: {df[(df['Method'] == 'flyp') & (df['Has_GradCLIP'])]['Avg OOD Accuracy'].max():.4f}")
print(f"   - Improvement: {((df[(df['Method'] == 'flyp') & (df['Has_GradCLIP'])]['Avg OOD Accuracy'].max() - df[df['Name'] == 'flyp_base_withText']['Avg OOD Accuracy'].values[0]) / df[df['Name'] == 'flyp_base_withText']['Avg OOD Accuracy'].values[0] * 100):.2f}%")

print("\n3. Max Gradient Norm Analysis:")
for method in ['carot', 'flyp']:
    gradclip_df = df[(df['Method'] == method) & (df['Has_GradCLIP'])]
    if len(gradclip_df) > 0:
        best_norm = gradclip_df.loc[gradclip_df['Avg OOD Accuracy'].idxmax()]['Max_Grad_Norm']
        print(f"   - {method.upper()} best max gradient norm for OOD: {best_norm}")

print("\n4. Trade-off Analysis:")
print("   - CAROT: Gradient clipping provides better OOD generalization while maintaining competitive ImageNet accuracy")
print("   - FLYP: Gradient clipping shows mixed results, with smaller improvements in OOD performance")

# Additional analysis plot - gradient norm vs performance improvement
fig2, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Max gradient norm vs OOD improvement
for method in ['carot', 'flyp']:
    base_model = f'{method}_base' if method == 'carot' else f'{method}_base_withText'
    base_ood = df[df['Name'] == base_model]['Avg OOD Accuracy'].values[0]
    
    gradclip_df = df[(df['Method'] == method) & (df['Has_GradCLIP'])]
    if len(gradclip_df) > 0:
        gradclip_df['OOD_Improvement'] = ((gradclip_df['Avg OOD Accuracy'] - base_ood) / base_ood * 100)
        
        # Group by max grad norm
        norm_grouped = gradclip_df.groupby('Max_Grad_Norm')['OOD_Improvement'].mean()
        
        ax[0].plot(norm_grouped.index, norm_grouped.values, 
                   marker='o', label=method.upper(), linewidth=2, markersize=8)

ax[0].set_xlabel('Max Gradient Norm', fontsize=12)
ax[0].set_ylabel('OOD Improvement (%)', fontsize=12)
ax[0].set_title('Gradient Clipping Threshold vs OOD Improvement', fontsize=14, fontweight='bold')
ax[0].set_xscale('log')
ax[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Plot 2: Scatter plot of all gradient clipping configurations
gradclip_all = df[df['Has_GradCLIP']]
colors = ['#FF6B6B' if 'carot' in name else '#4ECDC4' for name in gradclip_all['Name']]

scatter = ax[1].scatter(gradclip_all['Max_Grad_Norm'], gradclip_all['Avg OOD Accuracy'], 
                       c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)

ax[1].set_xlabel('Max Gradient Norm', fontsize=12)
ax[1].set_ylabel('Average OOD Accuracy', fontsize=12)
ax[1].set_title('Gradient Clipping Configurations', fontsize=14, fontweight='bold')
ax[1].set_xscale('log')
ax[1].grid(True, alpha=0.3)

# Add legend
carot_patch = mpatches.Patch(color='#FF6B6B', label='CAROT')
flyp_patch = mpatches.Patch(color='#4ECDC4', label='FLYP')
ax[1].legend(handles=[carot_patch, flyp_patch])

plt.tight_layout()
plt.savefig('gradient_norm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Add this code after the existing plots or as a standalone script

# Create a focused comparison plot for CAROT base vs best gradient clipping
fig3, ax = plt.subplots(1, 1, figsize=(12, 7))

# Get CAROT base results
carot_base_results = df[df['Name'] == 'carot_base'].iloc[0]

# Find best CAROT gradient clipping model based on Avg OOD Accuracy
carot_gradclip_models = df[(df['Method'] == 'carot') & (df['Has_GradCLIP'])]
best_carot_gradclip = carot_gradclip_models.loc[carot_gradclip_models['Avg OOD Accuracy'].idxmax()]

# Dataset names and columns
datasets = ['ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR', 'Avg OOD']
dataset_cols = ['ImageNet Accuracy', 'ImageNetV2 Accuracy', 'ImageNetSketch Accuracy', 
                'ImageNetA Accuracy', 'ImageNetR Accuracy', 'Avg OOD Accuracy']

# Extract values
base_values = [carot_base_results[col] for col in dataset_cols]
gradclip_values = [best_carot_gradclip[col] for col in dataset_cols]

# Calculate improvements
improvements = [(gradclip_values[i] - base_values[i]) / base_values[i] * 100 
                for i in range(len(base_values))]

# Set up bar positions
x = np.arange(len(datasets))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, base_values, width, label='CAROT Base', 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, gradclip_values, width, 
                label=f'CAROT Best GradCLIP ({best_carot_gradclip["Name"]})', 
                color='#FF6B6B', alpha=0.5, edgecolor='black', linewidth=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Add improvement percentages above the gradclip bars
for i, (bar, imp) in enumerate(zip(bars2, improvements)):
    height = bar.get_height()
    color = 'green' if imp > 0 else 'red'
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{imp:+.1f}%', ha='center', va='bottom', fontsize=9, 
            color=color, fontweight='bold')

# Customize plot
ax.set_xlabel('Dataset', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('CAROT: Base vs Best Gradient Clipping Performance Across All Datasets', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(max(base_values), max(gradclip_values)) * 1.15)

# Add a text box with summary
textstr = f'Best GradCLIP Config: {best_carot_gradclip["Name"]}\n'
textstr += f'Max Grad Norm: {best_carot_gradclip["Max_Grad_Norm"]}\n'
textstr += f'Avg OOD Improvement: {improvements[-1]:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('carot_base_vs_gradclip_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed comparison
print("\nDetailed CAROT Base vs Best Gradient Clipping Comparison:")
print("="*60)
print(f"Best Gradient Clipping Model: {best_carot_gradclip['Name']}")
print(f"Max Gradient Norm: {best_carot_gradclip['Max_Grad_Norm']}")
print("\nPerformance Comparison:")
print("-"*60)
print(f"{'Dataset':<20} {'Base':<10} {'GradCLIP':<10} {'Improvement':<15}")
print("-"*60)
for dataset, base_val, grad_val, imp in zip(datasets, base_values, gradclip_values, improvements):
    print(f"{dataset:<20} {base_val:<10.4f} {grad_val:<10.4f} {imp:+.2f}%")

# Also create a simplified version focusing only on OOD datasets
fig4, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# OOD datasets only
ood_datasets = ['ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR']
ood_dataset_cols = ['ImageNetV2 Accuracy', 'ImageNetSketch Accuracy', 
                    'ImageNetA Accuracy', 'ImageNetR Accuracy']

# Extract OOD values
ood_base_values = [carot_base_results[col] for col in ood_dataset_cols]
ood_gradclip_values = [best_carot_gradclip[col] for col in ood_dataset_cols]

# Calculate OOD improvements
ood_improvements = [(ood_gradclip_values[i] - ood_base_values[i]) / ood_base_values[i] * 100 
                    for i in range(len(ood_base_values))]

# Set up bar positions
x2 = np.arange(len(ood_datasets))
width2 = 0.35

# Create bars
bars3 = ax2.bar(x2 - width2/2, ood_base_values, width2, label='CAROT Base', 
                 color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
bars4 = ax2.bar(x2 + width2/2, ood_gradclip_values, width2, 
                 label='CAROT Best GradCLIP', 
                 color='#FF6B6B', alpha=0.5, edgecolor='black', linewidth=1)

# Add value labels and improvements
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Add improvement percentages
for i, (bar, imp) in enumerate(zip(bars4, ood_improvements)):
    height = bar.get_height()
    color = 'green' if imp > 0 else 'red'
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
             f'{imp:+.1f}%', ha='center', va='bottom', fontsize=10, 
             color=color, fontweight='bold')

# Draw horizontal line for average OOD accuracy
avg_ood_base = np.mean(ood_base_values)
avg_ood_gradclip = np.mean(ood_gradclip_values)
ax2.axhline(y=avg_ood_base, color='red', linestyle='--', alpha=0.5, 
            label=f'Avg OOD Base: {avg_ood_base:.3f}')
ax2.axhline(y=avg_ood_gradclip, color='green', linestyle='--', alpha=0.5, 
            label=f'Avg OOD GradCLIP: {avg_ood_gradclip:.3f}')

# Customize plot
ax2.set_xlabel('Out-of-Distribution Dataset', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=14)
ax2.set_title('CAROT: OOD Performance - Base vs Best Gradient Clipping', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x2)
ax2.set_xticklabels(ood_datasets)
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(max(ood_base_values), max(ood_gradclip_values)) * 1.15)

plt.tight_layout()
plt.savefig('carot_ood_comparison.png', dpi=300, bbox_inches='tight')
plt.show()