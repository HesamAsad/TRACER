import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Import all the necessary classes and functions from toy_experiment
from toy_experiment import (
    LightViT, LightTextTransformer, MultiModalContrastiveModel,
    MNISTMultiModal, ColoredMNISTMultiModal, tokenize_text,
    device, autocast, autocast_device, autocast_dtype,
    full_dataset, test_dataset, train_dataset, val_dataset,
    transform_rgb
)

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EmbeddingAnalyzer:
    """Class to analyze and visualize embeddings from different fine-tuning methods"""
    
    def __init__(self, checkpoint_dir="toy_exp_ckpts"):
        self.checkpoint_dir = checkpoint_dir
        self.models = {}
        self.embeddings = {}
        self.batch_size = 256
        
        # Initialize base model architecture
        self.image_encoder = LightViT().to(device)
        self.text_encoder = LightTextTransformer().to(device)
        
        # Create data loaders
        self._setup_data()
        
    def _setup_data(self):
        """Setup data loaders for analysis"""
        # Create multimodal datasets
        self.train_mm = MNISTMultiModal(train_dataset)
        self.test_mm = MNISTMultiModal(test_dataset)
        
        # Create colored datasets
        self.test_colored = ColoredMNISTMultiModal(test_dataset, color_shift=0)
        
        # Create data loaders - use smaller batch size for analysis
        self.test_loader = DataLoader(self.test_mm, batch_size=self.batch_size, shuffle=False)
        self.test_colored_loader = DataLoader(self.test_colored, batch_size=self.batch_size, shuffle=False)
        
    def load_models(self):
        """Load all fine-tuned models from checkpoints"""
        model_names = [
            'pretrained_multimodal',
            'finetuned_direct', 
            'finetuned_l2reg',
            'finetuned_selfdistill',
            'finetuned_dynamicdistill'
        ]
        
        print("Loading models...")
        for name in model_names:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pth")
            if os.path.exists(checkpoint_path):
                # Create fresh model instance
                image_enc = LightViT().to(device)
                text_enc = LightTextTransformer().to(device)
                model = MultiModalContrastiveModel(image_enc, text_enc).to(device)
                
                # Load checkpoint
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.eval()
                
                self.models[name] = model
                print(f"✓ Loaded {name}")
            else:
                print(f"✗ Checkpoint not found: {checkpoint_path}")
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def extract_embeddings(self, data_loader, max_samples=2000):
        """Extract embeddings from all models for a given data loader"""
        print(f"\nExtracting embeddings (max {max_samples} samples)...")
        
        embeddings = {name: {'image': [], 'text': []} for name in self.models.keys()}
        labels_list = []
        
        sample_count = 0
        for batch_idx, (images, texts, labels) in enumerate(tqdm(data_loader)):
            if sample_count >= max_samples:
                break
                
            images, texts = images.to(device), texts.to(device)
            labels_list.extend(labels.cpu().numpy())
            
            # Extract embeddings from all models
            with torch.no_grad():
                for name, model in self.models.items():
                    with autocast(device_type=autocast_device, dtype=autocast_dtype):
                        img_features = model.image_encoder(images, return_features=True)
                        txt_features = model.text_encoder(texts, return_features=True)
                    
                    embeddings[name]['image'].append(img_features.float().cpu().numpy())
                    embeddings[name]['text'].append(txt_features.float().cpu().numpy())
            
            sample_count += len(labels)
        
        # Concatenate all batches
        for name in embeddings.keys():
            embeddings[name]['image'] = np.concatenate(embeddings[name]['image'], axis=0)
            embeddings[name]['text'] = np.concatenate(embeddings[name]['text'], axis=0)
        
        return embeddings, np.array(labels_list[:sample_count])
    
    def reduce_dimensions(self, embeddings, method='tsne', n_components=2):
        """Apply dimensionality reduction to embeddings"""
        print(f"\nApplying {method.upper()} dimensionality reduction...")
        
        reduced_embeddings = {}
        
        for name in embeddings.keys():
            reduced_embeddings[name] = {}
            
            for modality in ['image', 'text']:
                data = embeddings[name][modality]
                
                if method == 'tsne':
                    reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
                elif method == 'pca':
                    reducer = PCA(n_components=n_components, random_state=42)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                reduced = reducer.fit_transform(data)
                reduced_embeddings[name][modality] = reduced
                
        return reduced_embeddings
    
    def compute_embedding_statistics(self, embeddings, labels):
        """Compute various statistics about the embeddings"""
        stats = {}
        
        for name in embeddings.keys():
            stats[name] = {}
            
            for modality in ['image', 'text']:
                data = embeddings[name][modality]
                
                # Within-class and between-class distances
                within_class_dists = []
                between_class_dists = []
                
                for label in range(10):  # 10 digits
                    class_mask = labels == label
                    class_data = data[class_mask]
                    other_data = data[~class_mask]
                    
                    if len(class_data) > 1:
                        # Within-class distances
                        class_sim = cosine_similarity(class_data)
                        within_class_dists.extend(class_sim[np.triu_indices_from(class_sim, k=1)])
                        
                        # Between-class distances (sample to avoid memory issues)
                        if len(other_data) > 100:
                            other_sample = other_data[np.random.choice(len(other_data), 100, replace=False)]
                        else:
                            other_sample = other_data
                        
                        between_sim = cosine_similarity(class_data, other_sample)
                        between_class_dists.extend(between_sim.flatten())
                
                stats[name][modality] = {
                    'within_class_similarity': np.mean(within_class_dists),
                    'between_class_similarity': np.mean(between_class_dists),
                    'separation_ratio': np.mean(within_class_dists) / np.mean(between_class_dists),
                    'embedding_norm': np.mean(np.linalg.norm(data, axis=1))
                }
        
        return stats
    
    def plot_embedding_space(self, reduced_embeddings, labels, save_path='embedding_spaces.png'):
        """Plot 2D embedding spaces for all models and modalities"""
        fig, axes = plt.subplots(2, len(self.models), figsize=(4*len(self.models), 8))
        
        if len(self.models) == 1:
            axes = axes.reshape(2, 1)
        
        model_names = list(self.models.keys())
        
        for j, name in enumerate(model_names):
            # Plot image embeddings
            scatter = axes[0, j].scatter(
                reduced_embeddings[name]['image'][:, 0],
                reduced_embeddings[name]['image'][:, 1],
                c=labels, cmap='tab10', alpha=0.6, s=15
            )
            axes[0, j].set_title(f'{name.replace("_", " ").title()}\nImage Embeddings')
            axes[0, j].set_xlabel('Dimension 1')
            axes[0, j].set_ylabel('Dimension 2')
            
            # Plot text embeddings
            axes[1, j].scatter(
                reduced_embeddings[name]['text'][:, 0],
                reduced_embeddings[name]['text'][:, 1],
                c=labels, cmap='tab10', alpha=0.6, s=15
            )
            axes[1, j].set_title(f'{name.replace("_", " ").title()}\nText Embeddings')
            axes[1, j].set_xlabel('Dimension 1')
            axes[1, j].set_ylabel('Dimension 2')
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes, label='Digit Class', shrink=0.8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_cross_modal_alignment(self, embeddings, labels, save_path='cross_modal_alignment.png'):
        """Analyze cross-modal alignment between image and text embeddings"""
        fig, axes = plt.subplots(2, len(self.models), figsize=(4*len(self.models), 8))
        
        if len(self.models) == 1:
            axes = axes.reshape(2, 1)
        
        model_names = list(self.models.keys())
        
        for j, name in enumerate(model_names):
            img_emb = embeddings[name]['image']
            txt_emb = embeddings[name]['text']
            
            # Compute cross-modal similarity matrix
            similarity = cosine_similarity(img_emb, txt_emb)
            
            # Plot similarity heatmap
            im1 = axes[0, j].imshow(similarity, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[0, j].set_title(f'{name.replace("_", " ").title()}\nCross-Modal Similarity')
            axes[0, j].set_xlabel('Text Samples')
            axes[0, j].set_ylabel('Image Samples')
            
            # Plot diagonal values (same sample similarity)
            diagonal_sim = np.diag(similarity)
            axes[1, j].hist(diagonal_sim, bins=30, alpha=0.7, density=True)
            axes[1, j].axvline(np.mean(diagonal_sim), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(diagonal_sim):.3f}')
            axes[1, j].set_title(f'Same-Sample Cross-Modal Similarity')
            axes[1, j].set_xlabel('Cosine Similarity')
            axes[1, j].set_ylabel('Density')
            axes[1, j].legend()
        
        plt.colorbar(im1, ax=axes[0, :], label='Cosine Similarity', shrink=0.8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_spurious_correlations(self, save_path='spurious_correlation_analysis.png'):
        """Analyze how different methods handle spurious correlations"""
        print("\nAnalyzing spurious correlations...")
        
        # Extract embeddings from colored dataset
        colored_embeddings, colored_labels = self.extract_embeddings(self.test_colored_loader, max_samples=1000)
        
        # Also get original embeddings for comparison
        orig_embeddings, orig_labels = self.extract_embeddings(self.test_loader, max_samples=1000)
        
        fig, axes = plt.subplots(2, len(self.models), figsize=(4*len(self.models), 8))
        if len(self.models) == 1:
            axes = axes.reshape(2, 1)
        
        model_names = list(self.models.keys())
        
        for j, name in enumerate(model_names):
            # Apply PCA for visualization
            pca_orig = PCA(n_components=2, random_state=42)
            pca_colored = PCA(n_components=2, random_state=42)
            
            orig_reduced = pca_orig.fit_transform(orig_embeddings[name]['image'])
            colored_reduced = pca_colored.fit_transform(colored_embeddings[name]['image'])
            
            # Plot original
            axes[0, j].scatter(orig_reduced[:, 0], orig_reduced[:, 1], 
                             c=orig_labels, cmap='tab10', alpha=0.6, s=15)
            axes[0, j].set_title(f'{name.replace("_", " ").title()}\nOriginal MNIST')
            
            # Plot colored
            axes[1, j].scatter(colored_reduced[:, 0], colored_reduced[:, 1], 
                             c=colored_labels, cmap='tab10', alpha=0.6, s=15)
            axes[1, j].set_title(f'Colored MNIST (Spurious)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return colored_embeddings, colored_labels
    
    def plot_embedding_statistics(self, stats, save_path='embedding_statistics.png'):
        """Plot embedding statistics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        model_names = list(stats.keys())
        modalities = ['image', 'text']
        
        metrics = ['within_class_similarity', 'between_class_similarity', 
                  'separation_ratio', 'embedding_norm']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            for modality in modalities:
                values = [stats[name][modality][metric] for name in model_names]
                x_pos = np.arange(len(model_names))
                if modality == 'image':
                    ax.bar(x_pos - 0.2, values, 0.4, label=f'{modality}', alpha=0.8)
                else:
                    ax.bar(x_pos + 0.2, values, 0.4, label=f'{modality}', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                               for name in model_names], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_analysis(self):
        """Run complete embedding analysis"""
        print("=" * 60)
        print("EMBEDDING SPACE ANALYSIS")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("No models loaded! Please check checkpoint directory.")
            return
        
        # Extract embeddings from test set
        embeddings, labels = self.extract_embeddings(self.test_loader, max_samples=4000)
        
        # Compute embedding statistics
        stats = self.compute_embedding_statistics(embeddings, labels)
        
        # Apply dimensionality reduction
        tsne_embeddings = self.reduce_dimensions(embeddings, method='tsne')
        pca_embeddings = self.reduce_dimensions(embeddings, method='pca')
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Plot embedding spaces
        self.plot_embedding_space(tsne_embeddings, labels, 'tsne_embedding_spaces.png')
        self.plot_embedding_space(pca_embeddings, labels, 'pca_embedding_spaces.png')
        
        # Plot cross-modal alignment
        self.plot_cross_modal_alignment(embeddings, labels, 'cross_modal_alignment.png')
        
        # Plot embedding statistics
        self.plot_embedding_statistics(stats, 'embedding_statistics.png')
        
        # Analyze spurious correlations
        colored_embeddings, colored_labels = self.analyze_spurious_correlations('spurious_correlation_analysis.png')
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("EMBEDDING STATISTICS SUMMARY")
        print("=" * 60)
        
        for name in stats.keys():
            print(f"\n{name.upper()}:")
            for modality in ['image', 'text']:
                print(f"  {modality.title()} Modality:")
                for metric, value in stats[name][modality].items():
                    print(f"    {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return embeddings, labels, stats

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(checkpoint_dir="/data/gpfs/projects/punim1316/CaRot/toy_exp_ckpts")
    
    # Run full analysis
    embeddings, labels, stats = analyzer.run_full_analysis()
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated visualization files:")
    print("- tsne_embedding_spaces.png")
    print("- pca_embedding_spaces.png") 
    print("- cross_modal_alignment.png")
    print("- embedding_statistics.png")
    print("- spurious_correlation_analysis.png")
    print("=" * 60)