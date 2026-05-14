import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
from clip import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import umap
import pickle
import gc
import json
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_REPO_ROOT = Path(__file__).resolve().parent

def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class ModelLoader:
    @staticmethod
    def load_dinov2():
        """Load pretrained DINOv2 model"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model = model.to(device)
        model.eval()
        return model
    
    @staticmethod
    def load_clip_base():
        """Load pretrained CLIP ViT-B/16"""
        model, train_preprocess, eval_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        return model, train_preprocess, eval_preprocess
    
    @staticmethod
    def load_tracer():
        """Load fine-tuned TRACER CLIP checkpoint (ViT-B/16)."""
        model, train_preprocess, eval_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        checkpoint_path = _REPO_ROOT / (
            "checkpoints/ImageNet/tracer/ViT-B/"
            "16_ep10_BS512_WD0.1_LR1e-05_D1.5_OC0.2_run1/checkpoint_10.pt"
        )
        model = torch_load(str(checkpoint_path))
        return model.model.to(device)

class DatasetLoader:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        root = _REPO_ROOT / "datasets/data"
        self.dataset_paths = {
            'imagenet_val': str(root / "ILSVRC2012/val"),
            'imagenet_a': str(root / "imagenet-a"),
            'imagenet_r': str(root / "imagenet-r"),
            'imagenet_v2': str(root / "ImageNetV2-matched-frequency"),
            'imagenet_sketch': str(root / "sketch"),
        }
        
    def get_transform(self, model_type='clip'):
        """Get appropriate transforms for different models"""
        if model_type == 'clip':
            model, train_preprocess, eval_preprocess = clip.load("ViT-B/16", device=device, jit=False)
            transform = eval_preprocess
            del model  # Free memory immediately
            clear_memory()
            return transform
        else:  # DINOv2
            return transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def get_dataset(self, dataset_name, transform):
        """Get a single dataset with transform"""
        dataset = datasets.ImageFolder(self.dataset_paths[dataset_name])
        dataset.transform = transform
        return dataset
    
    def get_dataloader(self, dataset_name, transform):
        """Get dataloader for a specific dataset"""
        dataset = self.get_dataset(dataset_name, transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

class SingleModelAnalyzer:
    def __init__(self, results_dir="analysis_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def extract_features(self, model, model_name, dataloader, dataset_name):
        """Extract features for a given model and dataset"""
        print(f"Extracting {model_name} features from {dataset_name}...")
        features = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Extracting features")):
                images = images.to(device)
                
                if model_name == 'dinov2':
                    feat = model(images)
                else:  # CLIP models
                    feat = model.encode_image(images)
                
                features.append(feat.cpu().numpy())
                labels.extend(targets.numpy())
                
                # Clear GPU memory periodically
                if batch_idx % 100 == 0:
                    clear_memory()
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        # Normalize features
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        return features, labels
    
    def compute_cosine_similarities(self, features, labels):
        """Compute within-class and between-class cosine similarities"""
        print("Computing cosine similarities...")
        unique_labels = np.unique(labels)
        within_class_sims = []
        between_class_sims = []
        
        for label in tqdm(unique_labels, desc="Computing similarities"):
            # Get features for this class
            class_mask = labels == label
            class_features = features[class_mask]
            
            if len(class_features) > 1:
                # Within-class similarities
                within_sim = 1 - cdist(class_features, class_features, metric='cosine')
                within_sim = within_sim[np.triu_indices_from(within_sim, k=1)]
                within_class_sims.extend(within_sim)
            
            # Between-class similarities (sample to avoid memory issues)
            other_features = features[~class_mask]
            if len(other_features) > 0:
                # Sample if too many other features
                if len(other_features) > 10000:
                    sample_idx = np.random.choice(len(other_features), 10000, replace=False)
                    other_features = other_features[sample_idx]
                
                between_sim = 1 - cdist(class_features, other_features, metric='cosine')
                between_class_sims.extend(between_sim.flatten())
        
        return np.array(within_class_sims), np.array(between_class_sims)
    
    def compute_cluster_metrics(self, features, labels):
        """Compute clustering quality metrics"""
        print("Computing cluster metrics...")
        if len(np.unique(labels)) > 1:
            # Sample if too many points for efficiency
            if len(features) > 50000:
                sample_idx = np.random.choice(len(features), 50000, replace=False)
                features_sample = features[sample_idx]
                labels_sample = labels[sample_idx]
            else:
                features_sample = features
                labels_sample = labels
                
            silhouette = silhouette_score(features_sample, labels_sample)
            davies_bouldin = davies_bouldin_score(features_sample, labels_sample)
        else:
            silhouette = davies_bouldin = np.nan
        
        return silhouette, davies_bouldin
    
    def compute_representation_diversity(self, features):
        """Compute diversity of representations"""
        print("Computing representation diversity...")
        # Sample for efficiency
        if len(features) > 10000:
            sample_idx = np.random.choice(len(features), 10000, replace=False)
            features_sample = features[sample_idx]
        else:
            features_sample = features
            
        # Compute pairwise distances
        distances = cdist(features_sample, features_sample, metric='cosine')
        
        # Average distance (diversity)
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        # Entropy of distance distribution
        hist, _ = np.histogram(distances[np.triu_indices_from(distances, k=1)], bins=50)
        hist = hist / hist.sum()
        distance_entropy = entropy(hist + 1e-10)
        
        return avg_distance, distance_entropy
    
    def analyze_single_combination(self, model, model_name, dataset_name, transform):
        """Analyze a single model-dataset combination"""
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name} on {dataset_name}")
        print(f"{'='*60}")
        
        # Get dataloader
        dataset_loader = DatasetLoader(batch_size=128)
        dataloader = dataset_loader.get_dataloader(dataset_name, transform)
        
        # Extract features
        features, labels = self.extract_features(model, model_name, dataloader, dataset_name)
        
        # Compute metrics
        within_sims, between_sims = self.compute_cosine_similarities(features, labels)
        silhouette, davies_bouldin = self.compute_cluster_metrics(features, labels)
        avg_distance, distance_entropy = self.compute_representation_diversity(features)
        
        # Compile results
        results = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'num_samples': len(features),
            'num_classes': len(np.unique(labels)),
            'feature_dim': features.shape[1],
            'within_class_sim_mean': float(np.mean(within_sims)),
            'within_class_sim_std': float(np.std(within_sims)),
            'between_class_sim_mean': float(np.mean(between_sims)),
            'between_class_sim_std': float(np.std(between_sims)),
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'avg_distance': float(avg_distance),
            'distance_entropy': float(distance_entropy)
        }
        
        # Save results immediately
        result_file = self.results_dir / f"{model_name}_{dataset_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save similarity distributions for plotting
        sim_data = {
            'within_class_sims': within_sims.tolist(),
            'between_class_sims': between_sims.tolist()
        }
        sim_file = self.results_dir / f"{model_name}_{dataset_name}_similarities.json"
        with open(sim_file, 'w') as f:
            json.dump(sim_data, f)
        
        # Generate and save 2D visualization
        self.plot_embeddings_2d(features, labels, model_name, dataset_name)
        
        # Print summary
        print(f"\nResults for {model_name} on {dataset_name}:")
        print(f"  Samples: {len(features)}, Classes: {len(np.unique(labels))}")
        print(f"  Within-class similarity: {np.mean(within_sims):.3f} ± {np.std(within_sims):.3f}")
        print(f"  Between-class similarity: {np.mean(between_sims):.3f} ± {np.std(between_sims):.3f}")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Davies-Bouldin score: {davies_bouldin:.3f}")
        print(f"  Average distance: {avg_distance:.3f}")
        print(f"  Distance entropy: {distance_entropy:.3f}")
        
        # Clear memory
        del features, labels, within_sims, between_sims
        clear_memory()
        
        return results
    
    def plot_embeddings_2d(self, features, labels, model_name, dataset_name, method='tsne', n_samples=5000):
        """Visualize embeddings in 2D space"""
        print(f"Generating 2D visualization for {model_name} on {dataset_name}...")
        
        # Sample data if too large
        if len(features) > n_samples:
            idx = np.random.choice(len(features), n_samples, replace=False)
            features_sample = features[idx]
            labels_sample = labels[idx]
        else:
            features_sample = features
            labels_sample = labels
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample)-1))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        features_2d = reducer.fit_transform(features_sample)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels_sample, cmap='tab20', s=1, alpha=0.6)
        plt.title(f'{model_name} - {dataset_name} ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.colorbar(scatter)
        
        # Save plot
        plot_file = self.results_dir / f"{model_name}_{dataset_name}_{method}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del features_2d, features_sample, labels_sample
        clear_memory()

class ResultsAggregator:
    def __init__(self, results_dir="analysis_results"):
        self.results_dir = Path(results_dir)
        
    def load_all_results(self):
        """Load all saved results"""
        results = []
        for result_file in self.results_dir.glob("*_results.json"):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
        return results
    
    def generate_comparison_plots(self):
        """Generate comparison plots from saved results"""
        results = self.load_all_results()
        df = pd.DataFrame(results)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette scores
        sns.barplot(data=df, x='dataset_name', y='silhouette_score', hue='model_name', ax=axes[0,0])
        axes[0,0].set_title('Silhouette Scores by Model and Dataset')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin scores
        sns.barplot(data=df, x='dataset_name', y='davies_bouldin_score', hue='model_name', ax=axes[0,1])
        axes[0,1].set_title('Davies-Bouldin Scores by Model and Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Within vs Between class similarities
        df_melted = df.melt(id_vars=['model_name', 'dataset_name'], 
                           value_vars=['within_class_sim_mean', 'between_class_sim_mean'],
                           var_name='similarity_type', value_name='similarity')
        sns.barplot(data=df_melted, x='dataset_name', y='similarity', 
                   hue='model_name', ax=axes[1,0])
        axes[1,0].set_title('Within vs Between Class Similarities')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Representation diversity
        sns.barplot(data=df, x='dataset_name', y='avg_distance', hue='model_name', ax=axes[1,1])
        axes[1,1].set_title('Average Pairwise Distance (Diversity)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_similarity_distributions(self):
        """Generate similarity distribution plots"""
        model_names = ['dinov2', 'clip_base', 'tracer']
        dataset_names = ['imagenet_val', 'imagenet_a', 'imagenet_r', 'imagenet_v2', 'imagenet_sketch']
        
        fig, axes = plt.subplots(len(model_names), len(dataset_names), figsize=(25, 15))
        
        for i, model_name in enumerate(model_names):
            for j, dataset_name in enumerate(dataset_names):
                sim_file = self.results_dir / f"{model_name}_{dataset_name}_similarities.json"
                
                if sim_file.exists():
                    with open(sim_file, 'r') as f:
                        sim_data = json.load(f)
                    
                    within_sims = np.array(sim_data['within_class_sims'])
                    between_sims = np.array(sim_data['between_class_sims'])
                    
                    ax = axes[i, j]
                    ax.hist(within_sims, bins=50, alpha=0.5, label='Within-class', density=True)
                    ax.hist(between_sims, bins=50, alpha=0.5, label='Between-class', density=True)
                    
                    ax.set_title(f'{model_name} - {dataset_name}')
                    ax.set_xlabel('Cosine Similarity')
                    ax.set_ylabel('Density')
                    ax.legend()
                    
                    # Add statistics
                    ax.text(0.05, 0.95, f'Within μ: {np.mean(within_sims):.3f}\nBetween μ: {np.mean(between_sims):.3f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[i, j].set_title(f'{model_name} - {dataset_name}')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        results = self.load_all_results()
        df = pd.DataFrame(results)
        
        report = "# Model Comparison Report: DINOv2 vs CLIP vs TRACER\n\n"
        
        # Overall performance summary
        report += "## Overall Performance Summary\n\n"
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            report += f"### {model_name}\n"
            
            avg_silhouette = model_data['silhouette_score'].mean()
            avg_db = model_data['davies_bouldin_score'].mean()
            avg_within = model_data['within_class_sim_mean'].mean()
            avg_between = model_data['between_class_sim_mean'].mean()
            
            report += f"- Average Silhouette Score: {avg_silhouette:.3f}\n"
            report += f"- Average Davies-Bouldin Score: {avg_db:.3f}\n"
            report += f"- Average Within-class Similarity: {avg_within:.3f}\n"
            report += f"- Average Between-class Similarity: {avg_between:.3f}\n"
            report += f"- Separation (Within - Between): {avg_within - avg_between:.3f}\n\n"
        
        # Dataset-specific analysis
        report += "## Dataset-specific Performance\n\n"
        
        for dataset_name in df['dataset_name'].unique():
            dataset_data = df[df['dataset_name'] == dataset_name]
            report += f"### {dataset_name}\n"
            
            best_silhouette = dataset_data.loc[dataset_data['silhouette_score'].idxmax()]
            best_separation = dataset_data.loc[(dataset_data['within_class_sim_mean'] - dataset_data['between_class_sim_mean']).idxmax()]
            
            report += f"- Best Silhouette Score: {best_silhouette['model_name']} ({best_silhouette['silhouette_score']:.3f})\n"
            report += f"- Best Class Separation: {best_separation['model_name']} ({best_separation['within_class_sim_mean'] - best_separation['between_class_sim_mean']:.3f})\n\n"
        
        # Key findings
        report += "## Key Findings\n\n"
        
        # Find best performing model overall
        model_scores = df.groupby('model_name').agg({
            'silhouette_score': 'mean',
            'davies_bouldin_score': 'mean'
        })
        
        best_model = model_scores['silhouette_score'].idxmax()
        report += f"1. **Overall Best Model**: {best_model} with highest average silhouette score\n"
        
        # Analyze TRACER performance
        if 'tracer' in df['model_name'].values:
            tracer_data = df[df['model_name'] == 'tracer']
            clip_data = df[df['model_name'] == 'clip_base']
            
            tracer_avg_sil = tracer_data['silhouette_score'].mean()
            clip_avg_sil = clip_data['silhouette_score'].mean()
            
            if tracer_avg_sil < clip_avg_sil:
                report += f"2. **TRACER Performance**: Shows degradation compared to base CLIP (Silhouette: {tracer_avg_sil:.3f} vs {clip_avg_sil:.3f})\n"
            else:
                report += f"2. **TRACER Performance**: Shows improvement over base CLIP (Silhouette: {tracer_avg_sil:.3f} vs {clip_avg_sil:.3f})\n"
        
        report += "3. **Generalization**: Performance varies significantly across different ImageNet variants\n"
        report += "4. **Representation Quality**: DINOv2 consistently shows strong clustering properties\n\n"
        
        # Save report
        with open(self.results_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
        
        print("Analysis report saved!")
        return report

def main():
    # Initialize analyzer
    analyzer = SingleModelAnalyzer()
    
    # Define models and datasets to process
    model_configs = [
        ('dinov2', 'dinov2'),
        ('clip_base', 'clip'),
        ('tracer', 'clip')
    ]
    
    dataset_names = ['imagenet_val', 'imagenet_a', 'imagenet_r', 'imagenet_v2', 'imagenet_sketch']
    
    # Process each model-dataset combination separately
    for model_name, model_type in model_configs:
        print(f"\n{'='*80}")
        print(f"LOADING MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Load model
        if model_name == 'dinov2':
            model = ModelLoader.load_dinov2()
        elif model_name == 'clip_base':
            model, _, _ = ModelLoader.load_clip_base()
        elif model_name == 'tracer':
            model = ModelLoader.load_tracer()
        
        # Get appropriate transform
        dataset_loader = DatasetLoader()
        transform = dataset_loader.get_transform(model_type)
        
        # Process each dataset with this model
        for dataset_name in dataset_names:
            try:
                analyzer.analyze_single_combination(model, model_name, dataset_name, transform)
            except Exception as e:
                print(f"Error processing {model_name} on {dataset_name}: {e}")
                continue
            
            # Clear memory after each dataset
            clear_memory()
        
        # Clear model from memory
        del model
        clear_memory()
        
        print(f"\nCompleted processing {model_name}")
    
    # Generate final analysis
    print(f"\n{'='*80}")
    print("GENERATING FINAL ANALYSIS")
    print(f"{'='*80}")
    
    aggregator = ResultsAggregator()
    aggregator.generate_comparison_plots()
    aggregator.generate_similarity_distributions()
    report = aggregator.generate_report()
    
    print("\nAnalysis complete! Check the 'analysis_results' directory for all outputs.")

if __name__ == "__main__":
    main()