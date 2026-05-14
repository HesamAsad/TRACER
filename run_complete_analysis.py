#!/usr/bin/env python3
"""
Complete Analysis Script for Fine-tuning Method Comparison

This script runs both embedding space analysis and decision boundary analysis
to provide a comprehensive comparison of different fine-tuning methods.

Usage:
    python run_complete_analysis.py

The script will generate multiple visualization files and a comprehensive report.
"""

import os
# Set OpenMP number of threads to 1 to avoid conflicts
os.environ['OMP_NUM_THREADS'] = '1'

# Set OpenBLAS to use single thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Set Intel MKL to use single thread (if available)
os.environ['MKL_NUM_THREADS'] = '1'

# Set VECLIB maximum threads (for macOS)
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Set NUMEXPR threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import argparse
from datetime import datetime

# Add current directory to path to import our analysis modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from embedding_analysis import EmbeddingAnalyzer
    from decision_boundary_analysis import DecisionBoundaryAnalyzer
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Make sure toy_experiment.py is in the same directory")
    sys.exit(1)

def create_analysis_report(embedding_stats, classification_results, 
                          separability_metrics, drift_metrics, 
                          output_file="analysis_report.md"):
    """Create a comprehensive markdown report of the analysis"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Fine-tuning Method Comparison Report

**Generated on:** {timestamp}

## Executive Summary

This report presents a comprehensive analysis comparing different fine-tuning strategies for multimodal contrastive learning models. The analysis includes:

1. **Embedding Space Analysis**: Visualization and statistical analysis of learned representations
2. **Decision Boundary Analysis**: Classification performance and separability metrics
3. **Feature Drift Analysis**: How representations change between original and spuriously correlated data

---

## 1. Embedding Statistics Summary

### 1.1 Within-class vs Between-class Similarity

| Method | Image Within-class | Image Between-class | Image Separation Ratio | Text Within-class | Text Between-class | Text Separation Ratio |
|--------|-------------------|-------------------|----------------------|------------------|------------------|---------------------|
"""
    
    # Add embedding statistics
    for method in embedding_stats.keys():
        img_stats = embedding_stats[method]['image']
        txt_stats = embedding_stats[method]['text']
        
        report += f"| {method.replace('_', ' ').title()} | "
        report += f"{img_stats['within_class_similarity']:.4f} | "
        report += f"{img_stats['between_class_similarity']:.4f} | "
        report += f"{img_stats['separation_ratio']:.4f} | "
        report += f"{txt_stats['within_class_similarity']:.4f} | "
        report += f"{txt_stats['between_class_similarity']:.4f} | "
        report += f"{txt_stats['separation_ratio']:.4f} |\n"
    
    report += "\n### 1.2 Embedding Norm Statistics\n\n"
    report += "| Method | Image Embedding Norm | Text Embedding Norm |\n"
    report += "|--------|---------------------|--------------------|\n"
    
    for method in embedding_stats.keys():
        img_norm = embedding_stats[method]['image']['embedding_norm']
        txt_norm = embedding_stats[method]['text']['embedding_norm']
        report += f"| {method.replace('_', ' ').title()} | {img_norm:.4f} | {txt_norm:.4f} |\n"
    
    # Add classification results
    report += "\n---\n\n## 2. Classification Performance\n\n"
    report += "### 2.1 Image Embedding Classification\n\n"
    report += "| Method | SVM | Logistic Regression | K-NN |\n"
    report += "|--------|-----|-------------------|------|\n"
    
    for method in classification_results.keys():
        img_results = classification_results[method]['image']
        report += f"| {method.replace('_', ' ').title()} | "
        report += f"{img_results['SVM']:.4f} | "
        report += f"{img_results['LogReg']:.4f} | "
        report += f"{img_results['KNN']:.4f} |\n"
    
    report += "\n### 2.2 Text Embedding Classification\n\n"
    report += "| Method | SVM | Logistic Regression | K-NN |\n"
    report += "|--------|-----|-------------------|------|\n"
    
    for method in classification_results.keys():
        txt_results = classification_results[method]['text']
        report += f"| {method.replace('_', ' ').title()} | "
        report += f"{txt_results['SVM']:.4f} | "
        report += f"{txt_results['LogReg']:.4f} | "
        report += f"{txt_results['KNN']:.4f} |\n"
    
    # Add separability metrics
    report += "\n---\n\n## 3. Embedding Separability Analysis\n\n"
    report += "| Method | Image Separability Ratio | Text Separability Ratio |\n"
    report += "|--------|--------------------------|------------------------|\n"
    
    for method in separability_metrics.keys():
        img_sep = separability_metrics[method]['image']['separability_ratio']
        txt_sep = separability_metrics[method]['text']['separability_ratio']
        report += f"| {method.replace('_', ' ').title()} | {img_sep:.4f} | {txt_sep:.4f} |\n"
    
    # Add drift analysis
    report += "\n---\n\n## 4. Feature Drift Analysis (Spurious Correlation Robustness)\n\n"
    report += "### 4.1 Mean Cosine Similarity (Original vs Colored)\n\n"
    report += "| Method | Image Similarity | Text Similarity |\n"
    report += "|--------|-----------------|-----------------|\n"
    
    for method in drift_metrics.keys():
        img_sim = drift_metrics[method]['image']['mean_similarity']
        txt_sim = drift_metrics[method]['text']['mean_similarity']
        report += f"| {method.replace('_', ' ').title()} | {img_sim:.4f} | {txt_sim:.4f} |\n"
    
    report += "\n### 4.2 Mean Class Centroid Drift\n\n"
    report += "| Method | Image Drift | Text Drift |\n"
    report += "|--------|------------|------------|\n"
    
    for method in drift_metrics.keys():
        img_drift = drift_metrics[method]['image']['mean_class_drift']
        txt_drift = drift_metrics[method]['text']['mean_class_drift']
        report += f"| {method.replace('_', ' ').title()} | {img_drift:.4f} | {txt_drift:.4f} |\n"
    
    # Add key insights
    report += "\n---\n\n## 5. Key Insights\n\n"
    
    # Find best performing methods
    best_img_classifier = max(classification_results.keys(), 
                             key=lambda x: classification_results[x]['image']['SVM'])
    best_txt_classifier = max(classification_results.keys(), 
                             key=lambda x: classification_results[x]['text']['SVM'])
    
    most_stable_img = max(drift_metrics.keys(), 
                         key=lambda x: drift_metrics[x]['image']['mean_similarity'])
    most_stable_txt = max(drift_metrics.keys(), 
                         key=lambda x: drift_metrics[x]['text']['mean_similarity'])
    
    report += f"### 5.1 Performance Summary\n\n"
    report += f"- **Best Image Classification**: {best_img_classifier.replace('_', ' ').title()}\n"
    report += f"- **Best Text Classification**: {best_txt_classifier.replace('_', ' ').title()}\n"
    report += f"- **Most Stable Image Embeddings**: {most_stable_img.replace('_', ' ').title()}\n"
    report += f"- **Most Stable Text Embeddings**: {most_stable_txt.replace('_', ' ').title()}\n"
    
    report += "\n### 5.2 Method-specific Observations\n\n"
    
    methods_info = {
        'pretrained_multimodal': 'Baseline pre-trained model without fine-tuning',
        'finetuned_direct': 'Direct fine-tuning with frozen text encoder',
        'finetuned_l2reg': 'L2 regularization to prevent drift from pre-trained weights',
        'finetuned_selfdistill': 'Self-distillation with static teacher',
        'finetuned_dynamicdistill': 'Dynamic self-distillation with EMA teacher'
    }
    
    for method, description in methods_info.items():
        if method in embedding_stats:
            report += f"**{method.replace('_', ' ').title()}**: {description}\n"
            
            # Add specific insights based on metrics
            img_sep = separability_metrics[method]['image']['separability_ratio']
            txt_sep = separability_metrics[method]['text']['separability_ratio']
            img_stability = drift_metrics[method]['image']['mean_similarity']
            
            if img_sep > 2.0:
                report += f"- Shows excellent image embedding separability (ratio: {img_sep:.3f})\n"
            elif img_sep > 1.5:
                report += f"- Shows good image embedding separability (ratio: {img_sep:.3f})\n"
            else:
                report += f"- Shows poor image embedding separability (ratio: {img_sep:.3f})\n"
                
            if img_stability > 0.8:
                report += f"- Highly robust to spurious correlations (similarity: {img_stability:.3f})\n"
            elif img_stability > 0.6:
                report += f"- Moderately robust to spurious correlations (similarity: {img_stability:.3f})\n"
            else:
                report += f"- Vulnerable to spurious correlations (similarity: {img_stability:.3f})\n"
            
            report += "\n"
    
    report += "\n---\n\n## 6. Generated Visualizations\n\n"
    report += "The following visualization files were generated:\n\n"
    report += "1. **tsne_embedding_spaces.png** - t-SNE visualization of embedding spaces\n"
    report += "2. **pca_embedding_spaces.png** - PCA visualization of embedding spaces\n"
    report += "3. **cross_modal_alignment.png** - Cross-modal similarity analysis\n"
    report += "4. **embedding_statistics.png** - Statistical comparison of embeddings\n"
    report += "5. **spurious_correlation_analysis.png** - Spurious correlation robustness\n"
    report += "6. **decision_boundaries_2d.png** - 2D decision boundary visualization\n"
    report += "7. **embedding_separability.png** - Separability metrics comparison\n"
    report += "8. **classification_performance.png** - Classification performance comparison\n"
    report += "9. **feature_drift.png** - Feature drift analysis\n"
    
    report += f"\n---\n\n*Report generated by Fine-tuning Analysis Suite on {timestamp}*\n"
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Comprehensive report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run complete fine-tuning analysis')
    parser.add_argument('--checkpoint_dir', default='/data/gpfs/projects/punim1316/CaRot/toy_exp_ckpts',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', default='./toy_experiment_figures_reproducible',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=4000,
                       help='Maximum number of samples for analysis')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        print("=" * 80)
        print("COMPREHENSIVE FINE-TUNING METHOD ANALYSIS")
        print("=" * 80)
        print(f"Checkpoint directory: {args.checkpoint_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Max samples: {args.max_samples}")
        print()
        
        # Run embedding analysis
        print("STEP 1: Running Embedding Space Analysis...")
        print("-" * 50)
        
        embedding_analyzer = EmbeddingAnalyzer(checkpoint_dir=args.checkpoint_dir)
        embeddings, labels, embedding_stats = embedding_analyzer.run_full_analysis()
        
        print("\nSTEP 2: Running Decision Boundary Analysis...")
        print("-" * 50)
        
        boundary_analyzer = DecisionBoundaryAnalyzer(checkpoint_dir=args.checkpoint_dir)
        classification_results, separability_metrics, drift_metrics = boundary_analyzer.run_full_analysis()
        
        print("\nSTEP 3: Generating Comprehensive Report...")
        print("-" * 50)
        
        # Generate comprehensive report
        create_analysis_report(embedding_stats, classification_results, 
                             separability_metrics, drift_metrics,
                             "comprehensive_analysis_report.md")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        
        # List all generated files
        generated_files = [
            "tsne_embedding_spaces.png",
            "pca_embedding_spaces.png", 
            "cross_modal_alignment.png",
            "embedding_statistics.png",
            "spurious_correlation_analysis.png",
            "decision_boundaries_2d.png",
            "embedding_separability.png",
            "classification_performance.png",
            "feature_drift.png",
            "comprehensive_analysis_report.md"
        ]
        
        for i, filename in enumerate(generated_files, 1):
            if os.path.exists(filename):
                print(f"{i:2d}. ✓ {filename}")
            else:
                print(f"{i:2d}. ✗ {filename} (not found)")
        
        print(f"\nAll files saved in: {os.path.abspath(output_dir)}")
        print("\nOpen 'comprehensive_analysis_report.md' for a detailed summary!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()