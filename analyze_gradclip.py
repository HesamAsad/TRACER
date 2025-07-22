import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read data from CSV file
df = pd.read_csv('wandb_export_2025-07-08T12_23_48.632+10_00.csv')

# Calculate average OOD accuracy where it's NaN
ood_columns = ['ImageNetSketch Accuracy', 'ImageNetA Accuracy', 'ImageNetR Accuracy'] # removed 'ImageNetV2 Accuracy'
df['Avg OOD Accuracy_calc'] = df[ood_columns].mean(axis=1)

# Use existing Avg OOD Acc if available, otherwise use calculated one
# df['Avg OOD Accuracy'] = df['Avg OOD Acc'].fillna(df['Avg OOD Accuracy_calc'])
df['Avg OOD Accuracy'] = df['Avg OOD Accuracy_calc']

# Extract coefficients
df['Self_Distil_Coef'] = pd.to_numeric(df['distil_coef'], errors='coerce')
df['Orth_Coef'] = pd.to_numeric(df['l_orth_wv'], errors='coerce').fillna(0)

# Categorize configurations
def categorize_config(row):
    name = row['Name']
    if name == 'carot':
        return 'Base'
    elif 'Layers' in name:
        return 'Layer-specific'
    elif 'beta_sd' in name:
        if row['Orth_Coef'] > 0:
            return 'Self-Distil + OC'
        else:
            return 'Self-Distillation'
    else:
        return 'Other'

df['Config_Type'] = df.apply(categorize_config, axis=1)

# Extract trainable layers for layer-specific configs
def extract_trainable_layers(name):
    if 'Layers' in name:
        parts = name.split('_')
        for part in parts:
            if 'Layers' in part:
                num = part.replace('Layers', '')
                try:
                    return int(num)
                except:
                    return None
    return None

df['Trainable_Layers'] = df['Name'].apply(extract_trainable_layers)

print("Configuration Analysis:")
print("=" * 50)
print(f"Total configurations: {len(df)}")
print(f"Base models: {len(df[df['Config_Type'] == 'Base'])}")
print(f"Self-distillation: {len(df[df['Config_Type'] == 'Self-Distillation'])}")
print(f"Self-distillation + OC: {len(df[df['Config_Type'] == 'Self-Distil + OC'])}")
print(f"Layer-specific fine-tuning: {len(df[df['Config_Type'] == 'Layer-specific'])}")
print()

# Print summary statistics
print("Summary Statistics:")
print("-" * 30)
for config_type in df['Config_Type'].unique():
    subset = df[df['Config_Type'] == config_type]
    print(f"{config_type}:")
    print(f"  ImageNet Acc: {subset['ImageNet Accuracy'].mean():.3f} ± {subset['ImageNet Accuracy'].std():.3f}")
    print(f"  Avg OOD Acc: {subset['Avg OOD Accuracy'].mean():.3f} ± {subset['Avg OOD Accuracy'].std():.3f}")
    print(f"  ImageNet ECE: {subset['ImageNet ECE'].mean():.3f} ± {subset['ImageNet ECE'].std():.3f}")
    print()

# 1. Enhanced Performance Heatmap
def create_enhanced_heatmap():
    # Remove empty spaces by shrinking figure to fit the data and increasing font sizes
    accuracy_columns = [
        'ImageNet Accuracy', 'ImageNetV2 Accuracy', 'ImageNetSketch Accuracy',
        'ImageNetA Accuracy', 'ImageNetR Accuracy', 'Avg OOD Accuracy'
    ]
    heatmap_data = df[['Name'] + accuracy_columns].set_index('Name')
    heatmap_data = heatmap_data.sort_values('Avg OOD Accuracy', ascending=False)

    # Set font sizes for better readability
    font_scale = 1.3
    sns.set(font_scale=font_scale, font="DejaVu Sans", style="whitegrid")
    plt.figure(figsize=(max(10, 0.7 * len(heatmap_data.columns)), min(0.6 * len(heatmap_data), 18)))

    # Create heatmap with larger annotation font and no whitespace
    g = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.65,
        vmin=0.45,
        vmax=0.85,
        cbar_kws={'label': 'Accuracy'},
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        linewidths=0.5,
        linecolor='gray',
        square=False
    )

    # Set axis label and tick font sizes
    g.set_title(
        'Performance Heatmap',
        fontsize=20, fontweight='bold', pad=18
    )
    g.set_xlabel('Datasets', fontsize=16, fontweight='bold', labelpad=12)
    g.set_ylabel('Configuration', fontsize=16, fontweight='bold', labelpad=12)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', fontsize=13, fontweight='bold')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=13, fontweight='bold')

    # Remove extra whitespace
    plt.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.08)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('enhanced_performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

# 2. Self-Distillation Analysis
def create_self_distillation_analysis():
    plt.figure(figsize=(15, 5))

    # Get self-distillation configs
    distil_configs = df[df['Config_Type'].isin(['Self-Distillation', 'Self-Distil + OC'])].copy()

    # Set font sizes
    plt.rc('axes', titlesize=15, labelsize=13)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    # Subplot 1: Self-Distillation Coefficient vs Accuracy
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(
        distil_configs['Self_Distil_Coef'], distil_configs['ImageNet Accuracy'],
        c=distil_configs['Orth_Coef'], cmap='viridis', s=100, alpha=0.7
    )
    plt.colorbar(scatter, label='Orthogonal Coef')
    plt.xlabel('Self-Distillation Coefficient', fontsize=13, fontweight='bold')
    plt.ylabel('ImageNet Accuracy', fontsize=13, fontweight='bold')
    plt.title('ID Performance vs Self-Distillation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Self-Distillation Coefficient vs OOD Accuracy
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(
        distil_configs['Self_Distil_Coef'], distil_configs['Avg OOD Accuracy'],
        c=distil_configs['Orth_Coef'], cmap='viridis', s=100, alpha=0.7
    )
    plt.colorbar(scatter, label='Orthogonal Coef')
    plt.xlabel('Self-Distillation Coefficient', fontsize=13, fontweight='bold')
    plt.ylabel('Average OOD Accuracy', fontsize=13, fontweight='bold')
    plt.title('OOD Performance vs Self-Distillation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Self-Distillation Coefficient vs ECE
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(
        distil_configs['Self_Distil_Coef'], distil_configs['ImageNet ECE'],
        c=distil_configs['Orth_Coef'], cmap='viridis', s=100, alpha=0.7
    )
    plt.colorbar(scatter, label='Orthogonal Coef')
    plt.xlabel('Self-Distillation Coefficient', fontsize=13, fontweight='bold')
    plt.ylabel('ImageNet ECE', fontsize=13, fontweight='bold')
    plt.title('Calibration vs Self-Distillation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('self_distillation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. ECE Analysis across datasets
def create_ece_analysis():
    plt.figure(figsize=(15, 5))

    ece_columns = ['ImageNet ECE', 'ImageNetA ECE', 'ImageNetR ECE', 'ImageNetSketch ECE', 'ImageNetV2 ECE']

    # Set font sizes
    plt.rc('axes', titlesize=15, labelsize=13)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    # Subplot 1: ECE Heatmap
    plt.subplot(1, 3, 1)
    ece_data = df[['Name'] + ece_columns].set_index('Name')
    ece_data = ece_data.sort_values('ImageNet ECE', ascending=True)

    g = sns.heatmap(
        ece_data, annot=True, fmt='.3f', cmap='RdYlBu',
        center=0.06, vmin=0.04, vmax=0.10, cbar_kws={'label': 'ECE'},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        linewidths=0.5, linecolor='gray', square=False
    )
    g.set_title('ECE Across Datasets', fontsize=14, fontweight='bold', pad=10)
    g.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    g.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', fontsize=11, fontweight='bold')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=11, fontweight='bold')
    plt.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.08)

    # Subplot 2: ID vs OOD Accuracy Trade-off
    plt.subplot(1, 3, 2)
    for config_type in df['Config_Type'].unique():
        subset = df[df['Config_Type'] == config_type]
        plt.scatter(
            subset['ImageNet Accuracy'], subset['Avg OOD Accuracy'],
            label=config_type, alpha=0.7, s=100
        )
    plt.xlabel('ImageNet Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Average OOD Accuracy', fontsize=12, fontweight='bold')
    plt.title('ID vs OOD Trade-off', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Accuracy vs ECE
    plt.subplot(1, 3, 3)
    for config_type in df['Config_Type'].unique():
        subset = df[df['Config_Type'] == config_type]
        plt.scatter(
            subset['ImageNet ECE'], subset['ImageNet Accuracy'],
            label=config_type, alpha=0.7, s=100
        )
    plt.xlabel('ImageNet ECE', fontsize=12, fontweight='bold')
    plt.ylabel('ImageNet Accuracy', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs Calibration', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ece_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Layer-specific analysis (if applicable)
def create_layer_analysis():
    layer_configs = df[df['Config_Type'] == 'Layer-specific']

    if len(layer_configs) > 0:
        plt.figure(figsize=(12, 4))

        # Set font sizes
        plt.rc('axes', titlesize=14, labelsize=12)
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        # Subplot 1: Layers vs Accuracy
        plt.subplot(1, 3, 1)
        plt.scatter(
            layer_configs['Trainable_Layers'], layer_configs['ImageNet Accuracy'],
            color='blue', alpha=0.7, s=100, label='ImageNet'
        )
        plt.scatter(
            layer_configs['Trainable_Layers'], layer_configs['Avg OOD Accuracy'],
            color='red', alpha=0.7, s=100, label='Avg OOD'
        )
        plt.xlabel('Number of Trainable Layers', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Layer-specific Fine-tuning', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Layers vs ECE
        plt.subplot(1, 3, 2)
        plt.scatter(
            layer_configs['Trainable_Layers'], layer_configs['ImageNet ECE'],
            color='green', alpha=0.7, s=100
        )
        plt.xlabel('Number of Trainable Layers', fontsize=12, fontweight='bold')
        plt.ylabel('ImageNet ECE', fontsize=12, fontweight='bold')
        plt.title('Calibration vs Layers', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Subplot 3: Parameter efficiency
        if 'trainable params' in layer_configs.columns:
            plt.subplot(1, 3, 3)
            plt.scatter(
                layer_configs['trainable params'], layer_configs['ImageNet Accuracy'],
                color='purple', alpha=0.7, s=100
            )
            plt.xlabel('Trainable Parameters', fontsize=12, fontweight='bold')
            plt.ylabel('ImageNet Accuracy', fontsize=12, fontweight='bold')
            plt.title('Parameter Efficiency', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

# Generate all plots
create_enhanced_heatmap()
create_self_distillation_analysis()
create_ece_analysis()
create_layer_analysis()

print("Analysis complete! Generated plots:")
print("- enhanced_performance_heatmap.png")
print("- self_distillation_analysis.png")
print("- ece_analysis.png")
print("- layer_analysis.png")

# Print best configurations
print("\nBest Configurations:")
print("-" * 40)
best_id = df.loc[df['ImageNet Accuracy'].idxmax()]
best_ood = df.loc[df['Avg OOD Accuracy'].idxmax()]
best_ece = df.loc[df['ImageNet ECE'].idxmin()]

print(f"Best ID Accuracy: {best_id['Name']} ({best_id['ImageNet Accuracy']:.3f})")
print(f"Best OOD Accuracy: {best_ood['Name']} ({best_ood['Avg OOD Accuracy']:.3f})")
print(f"Best Calibration: {best_ece['Name']} (ECE: {best_ece['ImageNet ECE']:.3f})")