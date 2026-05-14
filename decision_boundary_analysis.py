import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Import from toy_experiment
from toy_experiment import (
    LightViT, LightTextTransformer, MultiModalContrastiveModel,
    MNISTMultiModal, ColoredMNISTMultiModal,
    device, autocast, autocast_device, autocast_dtype,
    full_dataset, test_dataset, train_dataset, val_dataset
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionBoundaryAnalyzer:
    """Analyze decision boundaries and classification performance in embedding space"""
    
    def __init__(self, checkpoint_dir="toy_exp_ckpts"):
        self.checkpoint_dir = checkpoint_dir
        self.models = {}
        self.batch_size = 512
        
        # Setup data
        self._setup_data()
        
    def _setup_data(self):
        """Setup data loaders"""
        # Create datasets - using smaller subset for boundary analysis
        indices = np.random.choice(len(test_dataset), size=min(3000, len(test_dataset)), replace=False)
        test_subset = torch.utils.data.Subset(test_dataset, indices)
        
        self.test_mm = MNISTMultiModal(test_subset)
        self.test_colored = ColoredMNISTMultiModal(test_subset, color_shift=0)
        
        self.test_loader = DataLoader(self.test_mm, batch_size=self.batch_size, shuffle=False)
        self.test_colored_loader = DataLoader(self.test_colored, batch_size=self.batch_size, shuffle=False)
        
    def load_models(self):
        """Load fine-tuned models"""
        model_names = [
            'pretrained_multimodal',
            'finetuned_direct', 
            'finetuned_l2reg',
            'finetuned_selfdistill',
            'finetuned_dynamicdistill'
        ]
        
        print("Loading models for decision boundary analysis...")
        for name in model_names:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pth")
            if os.path.exists(checkpoint_path):
                # Create model
                image_enc = LightViT().to(device)
                text_enc = LightTextTransformer().to(device)
                model = MultiModalContrastiveModel(image_enc, text_enc).to(device)
                
                # Load weights
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.eval()
                
                self.models[name] = model
                print(f"✓ Loaded {name}")
        
        print(f"Loaded {len(self.models)} models")
    
    def extract_embeddings_with_split(self, data_loader, train_ratio=0.7):
        """Extract embeddings and split into train/test for boundary analysis"""
        all_embeddings = {name: {'image': [], 'text': []} for name in self.models.keys()}
        all_labels = []
        
        # Extract all embeddings
        with torch.no_grad():
            for images, texts, labels in tqdm(data_loader, desc="Extracting embeddings"):
                images, texts = images.to(device), texts.to(device)
                all_labels.extend(labels.cpu().numpy())
                
                for name, model in self.models.items():
                    with autocast(device_type=autocast_device, dtype=autocast_dtype):
                        img_features = model.image_encoder(images, return_features=True)
                        txt_features = model.text_encoder(texts, return_features=True)
                    
                    all_embeddings[name]['image'].append(img_features.float().cpu().numpy())
                    all_embeddings[name]['text'].append(txt_features.float().cpu().numpy())
        
        # Concatenate
        for name in all_embeddings.keys():
            all_embeddings[name]['image'] = np.concatenate(all_embeddings[name]['image'], axis=0)
            all_embeddings[name]['text'] = np.concatenate(all_embeddings[name]['text'], axis=0)
        
        labels = np.array(all_labels)
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, train_size=train_ratio, 
                                              stratify=labels, random_state=42)
        
        train_embeddings = {name: {'image': emb['image'][train_idx], 
                                  'text': emb['text'][train_idx]} 
                           for name, emb in all_embeddings.items()}
        test_embeddings = {name: {'image': emb['image'][test_idx], 
                                 'text': emb['text'][test_idx]} 
                          for name, emb in all_embeddings.items()}
        
        return (train_embeddings, test_embeddings, 
                labels[train_idx], labels[test_idx])
    
    def fit_classifiers(self, train_embeddings, train_labels):
        """Fit different classifiers to embedding spaces"""
        classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'LogReg': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        fitted_classifiers = {}
        
        for model_name in self.models.keys():
            fitted_classifiers[model_name] = {}
            
            for modality in ['image', 'text']:
                fitted_classifiers[model_name][modality] = {}
                X_train = train_embeddings[model_name][modality]
                
                print(f"Fitting classifiers for {model_name} - {modality}")
                
                for clf_name, clf in classifiers.items():
                    try:
                        clf_copy = type(clf)(**clf.get_params())
                        clf_copy.fit(X_train, train_labels)
                        fitted_classifiers[model_name][modality][clf_name] = clf_copy
                    except Exception as e:
                        print(f"Failed to fit {clf_name}: {e}")
                        fitted_classifiers[model_name][modality][clf_name] = None
        
        return fitted_classifiers
    
    def evaluate_classifiers(self, fitted_classifiers, test_embeddings, test_labels):
        """Evaluate classifier performance on embeddings"""
        results = {}
        
        for model_name in fitted_classifiers.keys():
            results[model_name] = {}
            
            for modality in ['image', 'text']:
                results[model_name][modality] = {}
                X_test = test_embeddings[model_name][modality]
                
                for clf_name, clf in fitted_classifiers[model_name][modality].items():
                    if clf is not None:
                        try:
                            y_pred = clf.predict(X_test)
                            accuracy = accuracy_score(test_labels, y_pred)
                            results[model_name][modality][clf_name] = accuracy
                        except Exception as e:
                            print(f"Evaluation failed for {clf_name}: {e}")
                            results[model_name][modality][clf_name] = 0.0
                    else:
                        results[model_name][modality][clf_name] = 0.0
        
        return results
    
    def plot_2d_decision_boundaries(self, embeddings, labels, save_path='decision_boundaries_2d.png'):
        """Plot 2D decision boundaries using PCA projection"""
        fig, axes = plt.subplots(len(self.models), 2, figsize=(12, 4*len(self.models)))
        
        if len(self.models) == 1:
            axes = axes.reshape(1, 2)
        
        model_names = list(self.models.keys())
        
        for i, model_name in enumerate(model_names):
            for j, modality in enumerate(['image', 'text']):
                ax = axes[i, j]
                
                # Apply PCA to reduce to 2D
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(embeddings[model_name][modality])
                
                # Fit SVM classifier for boundary visualization
                svm = SVC(kernel='rbf', random_state=42)
                svm.fit(X_2d, labels)
                
                # Create mesh for decision boundary
                h = 0.1  # step size
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                # Get decision boundary
                Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                ax.contour(xx, yy, Z, alpha=0.3, colors='black', linestyles='--', linewidths=0.5)
                
                # Plot points
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                                   cmap='tab10', alpha=0.7, s=20)
                
                ax.set_title(f'{model_name.replace("_", " ").title()}\n{modality.title()} Embeddings')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_embedding_separability(self, train_embeddings, train_labels, 
                                   test_embeddings, test_labels, 
                                   save_path='embedding_separability.png'):
        """Analyze and plot embedding separability across methods"""
        
        # Compute separability metrics
        separability_metrics = {}
        
        for model_name in self.models.keys():
            separability_metrics[model_name] = {}
            
            for modality in ['image', 'text']:
                X_train = train_embeddings[model_name][modality]
                X_test = test_embeddings[model_name][modality]
                
                # Compute intra-class and inter-class distances
                intra_dists = []
                inter_dists = []
                
                for digit in range(10):
                    # Intra-class distances (within same digit)
                    digit_mask = train_labels == digit
                    if np.sum(digit_mask) > 1:
                        digit_samples = X_train[digit_mask]
                        # Pairwise distances within class
                        from sklearn.metrics.pairwise import euclidean_distances
                        dist_matrix = euclidean_distances(digit_samples)
                        intra_dists.extend(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
                    
                    # Inter-class distances (to other digits)
                    other_mask = train_labels != digit
                    if np.sum(digit_mask) > 0 and np.sum(other_mask) > 0:
                        digit_samples = X_train[digit_mask]
                        other_samples = X_train[other_mask]
                        # Sample to avoid memory issues
                        if len(other_samples) > 500:
                            other_samples = other_samples[np.random.choice(len(other_samples), 500, replace=False)]
                        
                        inter_dist = euclidean_distances(digit_samples, other_samples)
                        inter_dists.extend(inter_dist.flatten())
                
                separability_metrics[model_name][modality] = {
                    'intra_class_distance': np.mean(intra_dists) if intra_dists else 0,
                    'inter_class_distance': np.mean(inter_dists) if inter_dists else 0,
                    'separability_ratio': np.mean(inter_dists) / np.mean(intra_dists) if intra_dists and inter_dists else 0
                }
        
        # Plot separability metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        model_names = list(self.models.keys())
        metrics = ['intra_class_distance', 'inter_class_distance', 'separability_ratio']
        metric_labels = ['Intra-class Distance', 'Inter-class Distance', 'Separability Ratio']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            for modality in ['image', 'text']:
                values = [separability_metrics[name][modality][metric] for name in model_names]
                x_pos = np.arange(len(model_names))
                
                if modality == 'image':
                    axes[i].bar(x_pos - 0.2, values, 0.4, label=modality, alpha=0.8)
                else:
                    axes[i].bar(x_pos + 0.2, values, 0.4, label=modality, alpha=0.8)
            
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(label)
            axes[i].set_title(label)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                                   for name in model_names], rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return separability_metrics
    
    def plot_classification_performance(self, results, save_path='classification_performance.png'):
        """Plot classification performance across different methods"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(results.keys())
        classifiers = ['SVM', 'LogReg', 'KNN']
        
        for j, modality in enumerate(['image', 'text']):
            ax = axes[j]
            
            x = np.arange(len(model_names))
            width = 0.25
            
            for i, clf in enumerate(classifiers):
                scores = [results[model][modality][clf] for model in model_names]
                ax.bar(x + i*width, scores, width, label=clf, alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{modality.title()} Embedding Classification Performance')
            ax.set_xticks(x + width)
            ax.set_xticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                               for name in model_names], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_drift(self, original_embeddings, colored_embeddings, 
                             original_labels, colored_labels, save_path='feature_drift.png'):
        """Analyze how embeddings change between original and colored datasets"""
        
        # Compute drift metrics
        drift_metrics = {}
        
        for model_name in self.models.keys():
            drift_metrics[model_name] = {}
            
            for modality in ['image', 'text']:
                orig_emb = original_embeddings[model_name][modality]
                color_emb = colored_embeddings[model_name][modality]
                
                # Ensure we have the same number of samples for comparison
                min_samples = min(len(orig_emb), len(color_emb))
                orig_emb = orig_emb[:min_samples]
                color_emb = color_emb[:min_samples]
                orig_labels_sub = original_labels[:min_samples]
                color_labels_sub = colored_labels[:min_samples]
                
                # Compute cosine similarity between corresponding samples
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = []
                
                for i in range(min_samples):
                    if orig_labels_sub[i] == color_labels_sub[i]:  # Same digit
                        sim = cosine_similarity(orig_emb[i:i+1], color_emb[i:i+1])[0, 0]
                        similarities.append(sim)
                
                # Per-class drift analysis
                class_drifts = {}
                for digit in range(10):
                    digit_orig = orig_emb[orig_labels_sub == digit]
                    digit_color = color_emb[color_labels_sub == digit]
                    
                    if len(digit_orig) > 0 and len(digit_color) > 0:
                        # Compute centroid shift
                        orig_centroid = np.mean(digit_orig, axis=0)
                        color_centroid = np.mean(digit_color, axis=0)
                        drift = np.linalg.norm(orig_centroid - color_centroid)
                        class_drifts[digit] = drift
                
                drift_metrics[model_name][modality] = {
                    'mean_similarity': np.mean(similarities) if similarities else 0,
                    'std_similarity': np.std(similarities) if similarities else 0,
                    'mean_class_drift': np.mean(list(class_drifts.values())) if class_drifts else 0,
                    'class_drifts': class_drifts
                }
        
        # Plot drift metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        model_names = list(self.models.keys())
        
        # Mean similarity plot
        for modality in ['image', 'text']:
            values = [drift_metrics[name][modality]['mean_similarity'] for name in model_names]
            x_pos = np.arange(len(model_names))
            
            if modality == 'image':
                axes[0, 0].bar(x_pos - 0.2, values, 0.4, label=modality, alpha=0.8)
            else:
                axes[0, 0].bar(x_pos + 0.2, values, 0.4, label=modality, alpha=0.8)
        
        axes[0, 0].set_title('Mean Cosine Similarity (Original vs Colored)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                                  for name in model_names], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean class drift plot
        for modality in ['image', 'text']:
            values = [drift_metrics[name][modality]['mean_class_drift'] for name in model_names]
            
            if modality == 'image':
                axes[0, 1].bar(x_pos - 0.2, values, 0.4, label=modality, alpha=0.8)
            else:
                axes[0, 1].bar(x_pos + 0.2, values, 0.4, label=modality, alpha=0.8)
        
        axes[0, 1].set_title('Mean Class Centroid Drift')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                                  for name in model_names], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-class drift heatmap for image modality
        class_drift_matrix = np.zeros((len(model_names), 10))
        for i, model_name in enumerate(model_names):
            for digit in range(10):
                if digit in drift_metrics[model_name]['image']['class_drifts']:
                    class_drift_matrix[i, digit] = drift_metrics[model_name]['image']['class_drifts'][digit]
        
        im1 = axes[1, 0].imshow(class_drift_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Per-Class Drift (Image Embeddings)')
        axes[1, 0].set_yticks(range(len(model_names)))
        axes[1, 0].set_yticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                                   for name in model_names])
        axes[1, 0].set_xticks(range(10))
        axes[1, 0].set_xticklabels([f'Digit {i}' for i in range(10)], rotation=45)
        plt.colorbar(im1, ax=axes[1, 0], label='Drift Magnitude')
        
        # Per-class drift heatmap for text modality
        class_drift_matrix_text = np.zeros((len(model_names), 10))
        for i, model_name in enumerate(model_names):
            for digit in range(10):
                if digit in drift_metrics[model_name]['text']['class_drifts']:
                    class_drift_matrix_text[i, digit] = drift_metrics[model_name]['text']['class_drifts'][digit]
        
        im2 = axes[1, 1].imshow(class_drift_matrix_text, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('Per-Class Drift (Text Embeddings)')
        axes[1, 1].set_yticks(range(len(model_names)))
        axes[1, 1].set_yticklabels([name.replace('_', ' ').replace('finetuned', 'FT').replace('pretrained', 'PT') 
                                   for name in model_names])
        axes[1, 1].set_xticks(range(10))
        axes[1, 1].set_xticklabels([f'Digit {i}' for i in range(10)], rotation=45)
        plt.colorbar(im2, ax=axes[1, 1], label='Drift Magnitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return drift_metrics
    
    def run_full_analysis(self):
        """Run complete decision boundary analysis"""
        print("=" * 60)
        print("DECISION BOUNDARY ANALYSIS")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("No models found!")
            return
        
        # Extract embeddings from original dataset
        print("\nExtracting embeddings from original dataset...")
        (train_embeddings, test_embeddings, 
         train_labels, test_labels) = self.extract_embeddings_with_split(self.test_loader)
        
        # Extract embeddings from colored dataset for drift analysis
        print("Extracting embeddings from colored dataset...")
        (colored_train_embeddings, colored_test_embeddings,
         colored_train_labels, colored_test_labels) = self.extract_embeddings_with_split(self.test_colored_loader)
        
        # Fit classifiers
        print("Fitting classifiers...")
        fitted_classifiers = self.fit_classifiers(train_embeddings, train_labels)
        
        # Evaluate classifiers
        print("Evaluating classifiers...")
        results = self.evaluate_classifiers(fitted_classifiers, test_embeddings, test_labels)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Plot 2D decision boundaries
        all_embeddings = {name: {'image': np.vstack([train_embeddings[name]['image'], test_embeddings[name]['image']]),
                                 'text': np.vstack([train_embeddings[name]['text'], test_embeddings[name]['text']])}
                         for name in self.models.keys()}
        all_labels = np.concatenate([train_labels, test_labels])
        
        self.plot_2d_decision_boundaries(all_embeddings, all_labels, 'decision_boundaries_2d.png')
        
        # Plot embedding separability
        separability_metrics = self.plot_embedding_separability(
            train_embeddings, train_labels, test_embeddings, test_labels, 
            'embedding_separability.png')
        
        # Plot classification performance
        self.plot_classification_performance(results, 'classification_performance.png')
        
        # Analyze feature drift
        drift_metrics = self.analyze_feature_drift(
            all_embeddings, 
            {name: {'image': np.vstack([colored_train_embeddings[name]['image'], colored_test_embeddings[name]['image']]),
                    'text': np.vstack([colored_train_embeddings[name]['text'], colored_test_embeddings[name]['text']])}
             for name in self.models.keys()},
            all_labels,
            np.concatenate([colored_train_labels, colored_test_labels]),
            'feature_drift.png')
        
        # Print summary
        print("\n" + "=" * 60)
        print("DECISION BOUNDARY ANALYSIS SUMMARY")
        print("=" * 60)
        
        print("\nClassification Performance (Accuracy):")
        for model_name in results.keys():
            print(f"\n{model_name.upper()}:")
            for modality in ['image', 'text']:
                print(f"  {modality.title()} Modality:")
                for clf_name, acc in results[model_name][modality].items():
                    print(f"    {clf_name}: {acc:.4f}")
        
        print("\nSeparability Metrics:")
        for model_name in separability_metrics.keys():
            print(f"\n{model_name.upper()}:")
            for modality in ['image', 'text']:
                metrics = separability_metrics[model_name][modality]
                print(f"  {modality.title()} Modality:")
                print(f"    Separability Ratio: {metrics['separability_ratio']:.4f}")
                print(f"    Intra-class Distance: {metrics['intra_class_distance']:.4f}")
                print(f"    Inter-class Distance: {metrics['inter_class_distance']:.4f}")
        
        return results, separability_metrics, drift_metrics

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DecisionBoundaryAnalyzer(checkpoint_dir="/data/gpfs/projects/punim1316/CaRot/toy_exp_ckpts")
    
    # Run analysis
    results, separability_metrics, drift_metrics = analyzer.run_full_analysis()
    
    print("\n" + "=" * 60)
    print("Analysis complete! Generated visualization files:")
    print("- decision_boundaries_2d.png")
    print("- embedding_separability.png")
    print("- classification_performance.png") 
    print("- feature_drift.png")
    print("=" * 60)