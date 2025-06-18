import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from io import StringIO

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create the data
data = """"Name","ImageNet Accuracy","ImageNetV2 Accuracy","ImageNetSketch Accuracy","ImageNetA Accuracy","ImageNetR Accuracy","max_grad_norm","ema_up_freq","grad_norm_multiplier","warmup_length","freeze_text_encoder","trainable_layers","Final Max Grad Norm","Initial Max Grad Norm","Scheduled Max Grad Norm","trainable params","ImageNet ECE","ImageNetA ECE","ImageNetR ECE","ImageNetSketch ECE","ImageNetV2 ECE","Max Grad Norm","Avg OOD Acc"
"carot_BMA_gradCLIP_0.0001","0.803","0.7208","0.5343","0.5464","0.7994",,,,,,,,,,,,,,,,,
"carot_EMA_gradCLIP_0.001","0.7989","0.716","0.5269","0.5349","0.7955",,,,,,,,,,,,,,,,,
"carot_EMA_gradCLIP_0.0005","0.7976","0.7155","0.5274","0.5383","0.797",,,,,,,,,,,,,,,,,
"carot_EMA_gradCLIP_0.0001","0.7904","0.7104","0.5266","0.5439","0.7966",,,,,,,,,,,,,,,,,
"carot_base","0.8302","0.7412","0.5295","0.5164","0.7786",,,,,,,,,,,,,,,,,
"flyp_ldreg_gradCLIP_0.0001","0.8186","0.7294","0.5034","0.5204","0.7367",,,,,,,,,,,,,,,,,
"flyp_ldreg_gradCLIP_0.00001","0.8065","0.7185","0.5076","0.5331","0.7599",,,,,,,,,,,,,,,,,
"flyp_gradCLIP_0.00001","0.8084","0.7224","0.5056","0.5315","0.7616",,,,,,,,,,,,,,,,,
"flyp_gradCLIP_0.0001","0.8196","0.7298","0.5018","0.5205","0.7306",,,,,,,,,,,,,,,,,
"flyp_base","0.8256","0.7294","0.4946","0.4804","0.7149",,,,,,,,,,,,,,,,,
"carot_6Layers","0.8091","0.7262","0.5403","0.5536","0.8044","0","0","0","500","false","6","","","","158","0.0823","0.0767","0.084","0.0824","0.056","",""
"carot_7Layers","0.8098","0.7256","0.539","0.5516","0.8029","0","0","0","500","false","7","","","","182","0.0807","0.0839","0.0833","0.0834","0.0568","",""
"carot_5Layers","0.808","0.7245","0.5386","0.5565","0.8038","0","0","0","500","false","5","","","","134","0.0829","0.0778","0.0818","0.0819","0.0588","",""
"carot_4Layers","0.8054","0.7239","0.5367","0.5537","0.8021","0","0","0","500","false","4","","","","110","0.0836","0.0755","0.0806","0.0801","0.0626","",""
"carot_3Layers","0.8008","0.7197","0.5315","0.5512","0.8003","0","0","0","500","false","3","","","","86","0.085","0.0732","0.0815","0.0793","0.0617","",""
"carot_2Layers","0.7939","0.7151","0.5267","0.5477","0.7981","0","0","0","500","false","2","","","","62","0.0868","0.072","0.0822","0.0798","0.0644","",""
"carot_justLastLayer","0.782","0.7029","0.5194","0.5413","0.7919","0","0","0","500","false","1","","","","","0.0896","0.0745","0.0843","0.0802","0.0655","",""
"carot_0.0001_0.01_sched","0.8139","0.7298","0.5346","0.5356","0.7967","0.0001","0","100","100","","","0.01","0.0001","0.009999999844027534","","0.0748","0.0827","0.0777","0.0804","0.0506","",""
"carot_0.00001_0.1_sched","0.8114","0.7279","0.5338","0.5261","0.7964","0.000001","0","100000","1500","","","0.1","0.000001","0.09999999842453627","","0.0759","0.09","0.0862","0.0862","0.0546","",""
"carot_0.00001_1_sched","0.8142","0.7308","0.5318","0.5244","0.7969","0.00001","0","100000","1500","","","1","0.00001","0.999999984245363","","0.0721","0.0948","0.0825","0.0827","0.0514","",""
"carot_0.0001_0.01_sched_moreLRwarmup","0.8063","0.7246","0.5326","0.5344","0.7983","0.00001","0","1000","2500","","","0.01","0.00001","0.009999999842609604","","0.0822","0.0772","0.0861","0.0857","0.0588","",""
"carot_0.00001_0.01_sched","0.8078","0.7253","0.5332","0.5345","0.798","0.00001","0","1000","500","","","0.01","0.00001","0.009999999842609604","","0.0817","0.0816","0.0834","0.0843","0.0576","",""
"carot_0.00001_0.001_sched","0.8008","0.7201","0.5339","0.5479","0.801","0.00001","0","100","500","","","0.001","0.00001","0.0009999999844027534","","0.0949","0.0713","0.092","0.0884","0.0697","",""
"carot_0.0001_0.01_sched","0.8114","0.7285","0.5346","0.5335","0.7977","0.0001","0","","500","","","0.01","0.0001","0.009999999844027534","","0.0774","0.0871","0.0809","0.083","0.0538","","""""

# Read data into DataFrame
df = pd.read_csv(StringIO(data))

# Calculate average OOD accuracy
df['Avg OOD Accuracy'] = df[['ImageNetV2 Accuracy', 'ImageNetSketch Accuracy', 
                              'ImageNetA Accuracy', 'ImageNetR Accuracy']].mean(axis=1)

# Extract method type and configuration details
df['Method'] = df['Name'].apply(lambda x: x.split('_')[0])

# Categorize configurations
def categorize_config(name):
    if 'base' in name:
        return 'Base'
    elif any(layer in name for layer in ['Layers', 'justLastLayer']):
        return 'Layer-specific'
    elif 'sched' in name:
        return 'Dynamic GradClip'
    elif 'gradCLIP' in name:
        return 'Static GradClip'
    else:
        return 'Other'

df['Config_Type'] = df['Name'].apply(categorize_config)

# Extract number of trainable layers for layer-specific configs
def extract_trainable_layers(name):
    if 'justLastLayer' in name:
        return 1
    elif any(f'{i}Layers' in name for i in range(2, 8)):
        for i in range(2, 8):
            if f'{i}Layers' in name:
                return i
    return None

df['Trainable_Layers'] = df['Name'].apply(extract_trainable_layers)

# For scheduled gradient clipping, extract final grad norm
def extract_final_grad_norm(row):
    if pd.notna(row['Final Max Grad Norm']) and row['Final Max Grad Norm'] != '':
        try:
            return float(row['Final Max Grad Norm'])
        except:
            return None
    return None

df['Final_Grad_Norm'] = df.apply(extract_final_grad_norm, axis=1)

# Extract static gradient clipping values
def extract_static_grad_norm(name):
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

df['Static_Grad_Norm'] = df['Name'].apply(extract_static_grad_norm)

print("Configuration Analysis:")
print("="*50)
print(f"Total configurations: {len(df)}")
print(f"Base models: {len(df[df['Config_Type'] == 'Base'])}")
print(f"Static gradient clipping: {len(df[df['Config_Type'] == 'Static GradClip'])}")
print(f"Dynamic gradient clipping: {len(df[df['Config_Type'] == 'Dynamic GradClip'])}")
print(f"Layer-specific fine-tuning: {len(df[df['Config_Type'] == 'Layer-specific'])}")

# =============================================================================
# Figure 1: Enhanced Performance Heatmap
# =============================================================================
def create_enhanced_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Prepare data for heatmap
    heatmap_data = df.set_index('Name')[['ImageNet Accuracy', 'ImageNetV2 Accuracy', 
                                          'ImageNetSketch Accuracy', 'ImageNetA Accuracy', 
                                          'ImageNetR Accuracy', 'Avg OOD Accuracy']]
    
    # Sort by configuration type and performance
    df_sorted = df.copy()
    df_sorted['Sort_Key'] = df_sorted.apply(lambda x: (
        0 if x['Config_Type'] == 'Base' else
        1 if x['Config_Type'] == 'Layer-specific' else
        2 if x['Config_Type'] == 'Static GradClip' else
        3 if x['Config_Type'] == 'Dynamic GradClip' else 4
    ), axis=1)
    df_sorted = df_sorted.sort_values(['Method', 'Sort_Key', 'Avg OOD Accuracy'], ascending=[True, True, False])
    
    heatmap_sorted = df_sorted.set_index('Name')[['ImageNet Accuracy', 'ImageNetV2 Accuracy', 
                                                   'ImageNetSketch Accuracy', 'ImageNetA Accuracy', 
                                                   'ImageNetR Accuracy', 'Avg OOD Accuracy']]
    
    # Create heatmap
    sns.heatmap(heatmap_sorted, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax, vmin=0.4, vmax=0.9,
                cbar_kws={'label': 'Accuracy'}, linewidths=0.8)
    
    # Add configuration type annotations
    y_pos = 0
    for method in ['carot', 'flyp']:
        method_data = df_sorted[df_sorted['Method'] == method]
        for config_type in ['Base', 'Layer-specific', 'Static GradClip', 'Dynamic GradClip']:
            config_data = method_data[method_data['Config_Type'] == config_type]
            if len(config_data) > 0:
                # Add colored rectangle to indicate configuration type
                colors = {'Base': '#FFE5CC', 'Layer-specific': '#CCE5FF', 
                         'Static GradClip': '#CCFFCC', 'Dynamic GradClip': '#FFCCFF'}
                rect = Rectangle((0, y_pos), heatmap_sorted.shape[1], len(config_data), 
                               linewidth=2, edgecolor=colors[config_type], facecolor='none')
                ax.add_patch(rect)
                y_pos += len(config_data)
    
    ax.set_title('Enhanced Performance Heatmap: All Configurations Across Datasets', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Datasets', fontsize=14)
    ax.set_ylabel('Model Configurations', fontsize=14)
    
    # Add legend for configuration types
    legend_elements = [
        mpatches.Patch(color='#FFE5CC', label='Base'),
        mpatches.Patch(color='#CCE5FF', label='Layer-specific'),
        mpatches.Patch(color='#CCFFCC', label='Static GradClip'),
        mpatches.Patch(color='#FFCCFF', label='Dynamic GradClip')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig('enhanced_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# Figure 2: Layer-specific Fine-tuning Analysis
# =============================================================================
def create_layer_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter layer-specific configurations
    layer_configs = df[df['Config_Type'] == 'Layer-specific'].copy()
    layer_configs = layer_configs.sort_values('Trainable_Layers')
    
    if len(layer_configs) > 0:
        # Plot 1: Performance vs Number of Trainable Layers
        ax1.plot(layer_configs['Trainable_Layers'], layer_configs['ImageNet Accuracy'], 
                'o-', label='ImageNet', linewidth=2, markersize=8, color='#2E8B57')
        ax1.plot(layer_configs['Trainable_Layers'], layer_configs['Avg OOD Accuracy'], 
                's--', label='Avg OOD', linewidth=2, markersize=8, color='#FF6347')
        
        ax1.set_xlabel('Number of Trainable Layers', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Performance vs Trainable Layers', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(layer_configs['Trainable_Layers'])
        
        # Plot 2: Trainable Parameters vs Performance
        # Extract trainable parameters where available
        layer_configs_with_params = layer_configs[layer_configs['trainable params'].notna() & 
                                                  (layer_configs['trainable params'] != '')]
        if len(layer_configs_with_params) > 0:
            trainable_params = layer_configs_with_params['trainable params'].astype(float)
            ax2.scatter(trainable_params, layer_configs_with_params['ImageNet Accuracy'], 
                       s=100, alpha=0.7, label='ImageNet', color='#2E8B57')
            ax2.scatter(trainable_params, layer_configs_with_params['Avg OOD Accuracy'], 
                       s=100, alpha=0.7, label='Avg OOD', color='#FF6347', marker='s')
            
            # Add labels for each point
            for _, row in layer_configs_with_params.iterrows():
                ax2.annotate(f"{int(row['Trainable_Layers'])}L", 
                           (row['trainable params'], row['ImageNet Accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('Trainable Parameters (M)', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Performance vs Trainable Parameters', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Trainable parameters data\nnot available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Performance vs Trainable Parameters', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('layer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# Figure 3: Gradient Clipping Comparison (Static vs Dynamic)
# =============================================================================
def create_gradient_clipping_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter gradient clipping configurations
    static_gradclip = df[df['Config_Type'] == 'Static GradClip'].copy()
    dynamic_gradclip = df[df['Config_Type'] == 'Dynamic GradClip'].copy()
    
    # Plot 1: Static Gradient Clipping
    if len(static_gradclip) > 0:
        for method in ['carot', 'flyp']:
            method_data = static_gradclip[static_gradclip['Method'] == method]
            if len(method_data) > 0:
                # Group by gradient norm
                grouped = method_data.groupby('Static_Grad_Norm').agg({
                    'ImageNet Accuracy': 'mean',
                    'Avg OOD Accuracy': 'mean'
                }).reset_index()
                
                ax1.plot(grouped['Static_Grad_Norm'], grouped['ImageNet Accuracy'], 
                        'o-', label=f'{method.upper()} ImageNet', linewidth=2, markersize=8)
                ax1.plot(grouped['Static_Grad_Norm'], grouped['Avg OOD Accuracy'], 
                        's--', label=f'{method.upper()} Avg OOD', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Static Gradient Norm', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Static Gradient Clipping Performance', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dynamic Gradient Clipping
    if len(dynamic_gradclip) > 0:
        # Sort by final gradient norm
        dynamic_gradclip_sorted = dynamic_gradclip.sort_values('Final_Grad_Norm')
        
        x_pos = np.arange(len(dynamic_gradclip_sorted))
        width = 0.35
        
        ax2.bar(x_pos - width/2, dynamic_gradclip_sorted['ImageNet Accuracy'], 
               width, label='ImageNet', alpha=0.8, color='#2E8B57')
        ax2.bar(x_pos + width/2, dynamic_gradclip_sorted['Avg OOD Accuracy'], 
               width, label='Avg OOD', alpha=0.8, color='#FF6347')
        
        # Create labels with initial and final grad norms
        labels = []
        for _, row in dynamic_gradclip_sorted.iterrows():
            initial = row['Initial Max Grad Norm'] if pd.notna(row['Initial Max Grad Norm']) else 'N/A'
            final = row['Final Max Grad Norm'] if pd.notna(row['Final Max Grad Norm']) else 'N/A'
            labels.append(f'{initial}→{final}')
        
        ax2.set_xlabel('Initial → Final Gradient Norm', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Dynamic Gradient Clipping Performance', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gradient_clipping_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# Figure 4: ImageNet vs OOD Performance Trade-off
# =============================================================================
def create_performance_tradeoff():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors and markers for different configuration types
    config_colors = {
        'Base': '#FF6B6B',
        'Layer-specific': '#4ECDC4', 
        'Static GradClip': '#45B7D1',
        'Dynamic GradClip': '#96CEB4'
    }
    
    config_markers = {
        'Base': 'o',
        'Layer-specific': 's',
        'Static GradClip': '^',
        'Dynamic GradClip': 'D'
    }
    
    # Plot each configuration type
    for config_type in config_colors.keys():
        config_data = df[df['Config_Type'] == config_type]
        if len(config_data) > 0:
            ax.scatter(config_data['ImageNet Accuracy'], config_data['Avg OOD Accuracy'],
                      c=config_colors[config_type], marker=config_markers[config_type],
                      s=120, alpha=0.7, edgecolors='black', linewidth=1,
                      label=config_type)
    
    # Add Pareto frontier
    pareto_indices = []
    sorted_df = df.sort_values('ImageNet Accuracy')
    max_ood = 0
    for idx, row in sorted_df.iterrows():
        if row['Avg OOD Accuracy'] >= max_ood:
            pareto_indices.append(idx)
            max_ood = row['Avg OOD Accuracy']
    
    pareto_df = df.loc[pareto_indices].sort_values('ImageNet Accuracy')
    ax.plot(pareto_df['ImageNet Accuracy'], pareto_df['Avg OOD Accuracy'], 
           'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    # Annotate best models
    best_imagenet = df.loc[df['ImageNet Accuracy'].idxmax()]
    best_ood = df.loc[df['Avg OOD Accuracy'].idxmax()]
    
    ax.annotate(f"Best ImageNet\n{best_imagenet['Name']}", 
               (best_imagenet['ImageNet Accuracy'], best_imagenet['Avg OOD Accuracy']),
               xytext=(10, 10), textcoords='offset points', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate(f"Best OOD\n{best_ood['Name']}", 
               (best_ood['ImageNet Accuracy'], best_ood['Avg OOD Accuracy']),
               xytext=(-10, -20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('ImageNet Accuracy', fontsize=14)
    ax.set_ylabel('Average OOD Accuracy', fontsize=14)
    ax.set_title('ImageNet vs OOD Performance Trade-off Across All Configurations', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# Figure 5: Configuration Type Summary
# =============================================================================
def create_configuration_summary():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average performance by configuration type
    config_summary = df.groupby('Config_Type').agg({
        'ImageNet Accuracy': ['mean', 'std'],
        'Avg OOD Accuracy': ['mean', 'std']
    }).round(4)
    
    config_types = config_summary.index
    imagenet_means = config_summary[('ImageNet Accuracy', 'mean')]
    imagenet_stds = config_summary[('ImageNet Accuracy', 'std')]
    ood_means = config_summary[('Avg OOD Accuracy', 'mean')]
    ood_stds = config_summary[('Avg OOD Accuracy', 'std')]
    
    x = np.arange(len(config_types))
    width = 0.35
    
    ax1.bar(x - width/2, imagenet_means, width, yerr=imagenet_stds, 
           label='ImageNet', alpha=0.8, capsize=5, color='#2E8B57')
    ax1.bar(x + width/2, ood_means, width, yerr=ood_stds, 
           label='Avg OOD', alpha=0.8, capsize=5, color='#FF6347')
    
    ax1.set_xlabel('Configuration Type', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Average Performance by Configuration Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Best performance by configuration type
    best_configs = df.loc[df.groupby('Config_Type')['Avg OOD Accuracy'].idxmax()]
    
    ax2.scatter(best_configs['ImageNet Accuracy'], best_configs['Avg OOD Accuracy'],
               s=200, alpha=0.7, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    for _, row in best_configs.iterrows():
        ax2.annotate(row['Config_Type'], 
                    (row['ImageNet Accuracy'], row['Avg OOD Accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('ImageNet Accuracy', fontsize=12)
    ax2.set_ylabel('Average OOD Accuracy', fontsize=12)
    ax2.set_title('Best Model from Each Configuration Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of configurations by type
    config_counts = df['Config_Type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax3.pie(config_counts.values, labels=config_counts.index, 
                                      autopct='%1.1f%%', colors=colors[:len(config_counts)])
    ax3.set_title('Distribution of Configuration Types', fontsize=14, fontweight='bold')
    
    # Plot 4: Improvement over base models
    base_models = df[df['Config_Type'] == 'Base']
    improvements = []
    improvement_labels = []
    
    for method in ['carot', 'flyp']:
        base_model = base_models[base_models['Method'] == method]
        if len(base_model) > 0:
            base_ood = base_model['Avg OOD Accuracy'].iloc[0]
            method_configs = df[df['Method'] == method]
            
            for config_type in ['Layer-specific', 'Static GradClip', 'Dynamic GradClip']:
                config_data = method_configs[method_configs['Config_Type'] == config_type]
                if len(config_data) > 0:
                    best_config = config_data.loc[config_data['Avg OOD Accuracy'].idxmax()]
                    improvement = ((best_config['Avg OOD Accuracy'] - base_ood) / base_ood) * 100
                    improvements.append(improvement)
                    improvement_labels.append(f'{method.upper()}\n{config_type}')
    
    if improvements:
        colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax4.bar(improvement_labels, improvements, color=colors_imp, alpha=0.7)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('Improvement over Base (%)', fontsize=12)
        ax4.set_title('Best Configuration Improvement over Base', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('configuration_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Generate all figures
create_enhanced_heatmap()
create_layer_analysis()
create_gradient_clipping_comparison()
create_performance_tradeoff()
create_configuration_summary()

# Print detailed analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS RESULTS")
print("="*80)

print(f"\n1. BEST OVERALL PERFORMANCE:")
best_imagenet = df.loc[df['ImageNet Accuracy'].idxmax()]
best_ood = df.loc[df['Avg OOD Accuracy'].idxmax()]
print(f"   Best ImageNet: {best_imagenet['Name']} ({best_imagenet['ImageNet Accuracy']:.4f})")
print(f"   Best OOD: {best_ood['Name']} ({best_ood['Avg OOD Accuracy']:.4f})")

print(f"\n2. CONFIGURATION TYPE ANALYSIS:")
for config_type in df['Config_Type'].unique():
    config_data = df[df['Config_Type'] == config_type]
    if len(config_data) > 0:
        print(f"   {config_type}:")
        print(f"     Count: {len(config_data)}")
        print(f"     Avg ImageNet: {config_data['ImageNet Accuracy'].mean():.4f} ± {config_data['ImageNet Accuracy'].std():.4f}")
        print(f"     Avg OOD: {config_data['Avg OOD Accuracy'].mean():.4f} ± {config_data['Avg OOD Accuracy'].std():.4f}")
        best_in_type = config_data.loc[config_data['Avg OOD Accuracy'].idxmax()]
        print(f"     Best: {best_in_type['Name']} (OOD: {best_in_type['Avg OOD Accuracy']:.4f})")

print(f"\n3. LAYER-SPECIFIC FINE-TUNING INSIGHTS:")
layer_configs = df[df['Config_Type'] == 'Layer-specific']
if len(layer_configs) > 0:
    print(f"   Range: {layer_configs['Trainable_Layers'].min()}-{layer_configs['Trainable_Layers'].max()} layers")
    best_layer_config = layer_configs.loc[layer_configs['Avg OOD Accuracy'].idxmax()]
    print(f"   Best layer config: {best_layer_config['Trainable_Layers']} layers")
    print(f"   Performance: ImageNet {best_layer_config['ImageNet Accuracy']:.4f}, OOD {best_layer_config['Avg OOD Accuracy']:.4f}")

print(f"\n4. GRADIENT CLIPPING INSIGHTS:")
static_configs = df[df['Config_Type'] == 'Static GradClip']
dynamic_configs = df[df['Config_Type'] == 'Dynamic GradClip']

if len(static_configs) > 0:
    best_static = static_configs.loc[static_configs['Avg OOD Accuracy'].idxmax()]
    print(f"   Best static GradClip: {best_static['Name']}")
    print(f"   Gradient norm: {best_static['Static_Grad_Norm']}")
    print(f"   Performance: ImageNet {best_static['ImageNet Accuracy']:.4f}, OOD {best_static['Avg OOD Accuracy']:.4f}")

if len(dynamic_configs) > 0:
    best_dynamic = dynamic_configs.loc[dynamic_configs['Avg OOD Accuracy'].idxmax()]
    print(f"   Best dynamic GradClip: {best_dynamic['Name']}")
    print(f"   Performance: ImageNet {best_dynamic['ImageNet Accuracy']:.4f}, OOD {best_dynamic['Avg OOD Accuracy']:.4f}")

print(f"\n5. METHOD COMPARISON (CAROT vs FLYP):")
for method in ['carot', 'flyp']:
    method_data = df[df['Method'] == method]
    print(f"   {method.upper()}:")
    print(f"     Total configs: {len(method_data)}")
    print(f"     Best ImageNet: {method_data['ImageNet Accuracy'].max():.4f}")
    print(f"     Best OOD: {method_data['Avg OOD Accuracy'].max():.4f}")
    best_overall = method_data.loc[method_data['Avg OOD Accuracy'].idxmax()]
    print(f"     Best overall: {best_overall['Name']}")

print("\nAll analysis figures have been saved as separate PNG files:")
print("- enhanced_performance_heatmap.png")
print("- layer_analysis.png") 
print("- gradient_clipping_comparison.png")
print("- performance_tradeoff.png")
print("- configuration_summary.png")