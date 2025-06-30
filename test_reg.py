import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from clip import clip
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SyntheticCLIPExperiment:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load pretrained CLIP
        self.clip_model, self.train_preprocess, self.test_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        self.feature_dim = 512
        
        # Freeze CLIP and extract components
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Extract visual encoder and projection
        self.visual_encoder = self.clip_model.visual
        self.visual_projection = self.clip_model.visual.proj
        
    def generate_synthetic_data(self, n_samples=1000, n_classes=10, shift_magnitude=0.3):
        """Generate synthetic ID and OOD data with controlled distribution shift."""
        torch.manual_seed(42)
        
        # Generate base features on hypersphere
        base_features = torch.randn(n_samples, self.feature_dim)
        base_features = F.normalize(base_features, p=2, dim=1)
        
        # Create class structure
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Add class-specific patterns
        class_centers = F.normalize(torch.randn(n_classes, self.feature_dim), p=2, dim=1)
        
        id_features = []
        ood_features = []
        
        for i in range(n_samples):
            class_idx = labels[i]
            # ID: concentrated around class centers
            noise = torch.randn(self.feature_dim) * 0.3
            id_feat = base_features[i] + class_centers[class_idx] + noise
            id_features.append(F.normalize(id_feat, p=2, dim=0))
            
            # OOD: shifted distribution
            shift = torch.randn(self.feature_dim) * shift_magnitude
            ood_feat = base_features[i] + class_centers[class_idx] + noise + shift
            ood_features.append(F.normalize(ood_feat, p=2, dim=0))
        
        id_features = torch.stack(id_features)
        ood_features = torch.stack(ood_features)
        
        # Create text features (simplified - using class prototypes)
        text_features = F.normalize(class_centers + torch.randn_like(class_centers) * 0.1, p=2, dim=1)
        
        return {
            'id_features': id_features.to(self.device),
            'ood_features': ood_features.to(self.device),
            'text_features': text_features.to(self.device),
            'labels': labels.to(self.device)
        }
    
    def compute_metrics(self, features, labels, text_features):
        """Compute comprehensive metrics for evaluation."""
        # Singular value analysis
        _, S, _ = torch.svd(features)
        
        # Normalized covariance
        features_centered = features - features.mean(dim=0, keepdim=True)
        cov = torch.mm(features_centered.t(), features_centered) / features.size(0)
        eigenvalues = torch.linalg.eigvalsh(cov)
        
        # Effective rank
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy)
        
        # Calibration (simplified ECE)
        # Compute similarities and predictions
        similarities = torch.mm(features, text_features.t())
        probs = F.softmax(similarities, dim=1)
        predictions = probs.argmax(dim=1)
        
        # Compute confidence and accuracy per bin
        confidences, _ = probs.max(dim=1)
        accuracies = (predictions == labels).float()
        
        # ECE calculation
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(self.device)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
                ece += (in_bin.sum().float() / features.size(0)) * torch.abs(bin_confidence - bin_accuracy)
        
        metrics = {
            'sigma_min': S.min().item(),
            'sigma_max': S.max().item(),
            'condition_number': (S.max() / S.min()).item(),
            'effective_rank': effective_rank.item(),
            'accuracy': accuracies.mean().item(),
            'ece': ece.item(),
            'mean_confidence': confidences.mean().item(),
            'eigenvalue_sum': eigenvalues.sum().item()
        }
        
        return metrics, S, eigenvalues
    
    def apply_regularization(self, features, text_features, reg_type, reg_strength, 
                           pretrained_features=None, projection_matrix=None):
        """Apply specified regularization and return modified features."""
        
        if reg_type == 'none':
            return features
        
        # Create a simple model to optimize features
        optimized_features = nn.Parameter(features.clone())
        optimizer = torch.optim.Adam([optimized_features], lr=0.01)
        
        for _ in range(100):  # Optimization steps
            optimizer.zero_grad()
            
            # Normalize features
            norm_features = F.normalize(optimized_features, p=2, dim=1)
            
            # Task loss (simplified contrastive)
            similarities = torch.mm(norm_features, text_features.t()) / 0.07
            labels = torch.arange(min(norm_features.size(0), text_features.size(0))).to(self.device)
            task_loss = F.cross_entropy(similarities[:len(labels)], labels)
            
            # Regularization loss
            reg_loss = 0.0
            
            if reg_type == 'geodesic' and pretrained_features is not None:
                cos_sim = (norm_features * pretrained_features).sum(dim=1)
                cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
                reg_loss = torch.arccos(cos_sim).pow(2).mean()
                
            elif reg_type == 'spectral':
                cov = torch.mm(norm_features.t(), norm_features) / norm_features.size(0)
                eigenvalues = torch.linalg.eigvalsh(cov + 1e-5 * torch.eye(cov.size(0)).to(self.device))
                reg_loss = -torch.sum(torch.log(eigenvalues + 1e-5))
                
            elif reg_type == 'vmf':
                mean_direction = norm_features.mean(dim=0)
                r_bar = torch.norm(mean_direction)
                d = norm_features.size(1)
                kappa = r_bar * (d - r_bar**2) / (1 - r_bar**2 + 1e-7)
                reg_loss = kappa * r_bar
                
            elif reg_type == 'orthogonal' and projection_matrix is not None:
                # Apply to projection matrix
                gram = torch.mm(projection_matrix.t(), projection_matrix)
                I = torch.eye(gram.size(0)).to(self.device)
                reg_loss = torch.norm(gram - I, p='fro')**2
                
            elif reg_type == 'grassmann' and pretrained_features is not None:
                # Simplified - just use top components
                U1, _, _ = torch.svd(norm_features.t())
                U2, _, _ = torch.svd(pretrained_features.t())
                k = min(50, U1.size(1), U2.size(1))
                M = torch.mm(U1[:, :k].t(), U2[:, :k])
                _, S, _ = torch.svd(M)
                S_clamped = torch.clamp(S, -1 + 1e-7, 1 - 1e-7)
                principal_angles = torch.arccos(S_clamped)
                reg_loss = torch.sum(principal_angles**2)
            
            # Total loss
            total_loss = task_loss + reg_strength * reg_loss
            total_loss.backward()
            optimizer.step()
        
        return F.normalize(optimized_features.detach(), p=2, dim=1)
    
    def visualize_results(self, results_dict, save_path='regularization_effects.png'):
        """Create comprehensive visualization of regularization effects."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Singular Value Spectrum
        ax1 = plt.subplot(3, 3, 1)
        for name, data in results_dict.items():
            S = data['singular_values']
            ax1.semilogy(S.cpu().numpy()[:100], label=name, linewidth=2)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value (log scale)')
        ax1.set_title('Singular Value Spectrum')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Condition Number Comparison
        ax2 = plt.subplot(3, 3, 2)
        methods = list(results_dict.keys())
        condition_numbers = [results_dict[m]['metrics']['condition_number'] for m in methods]
        bars = ax2.bar(methods, condition_numbers)
        ax2.set_ylabel('Condition Number')
        ax2.set_title('Condition Number (Lower is Better)')
        ax2.set_yscale('log')
        # Color bars based on value
        for bar, cn in zip(bars, condition_numbers):
            bar.set_color(plt.cm.RdYlGn_r(np.log10(cn) / 4))
        
        # 3. Effective Rank
        ax3 = plt.subplot(3, 3, 3)
        effective_ranks = [results_dict[m]['metrics']['effective_rank'] for m in methods]
        bars = ax3.bar(methods, effective_ranks)
        ax3.set_ylabel('Effective Rank')
        ax3.set_title('Effective Rank (Higher is Better)')
        for bar, er in zip(bars, effective_ranks):
            bar.set_color(plt.cm.RdYlGn(er / max(effective_ranks)))
        
        # 4. ECE Comparison
        ax4 = plt.subplot(3, 3, 4)
        id_ece = [results_dict[m]['metrics']['ece'] for m in methods]
        ood_ece = [results_dict[m]['ood_metrics']['ece'] for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        ax4.bar(x - width/2, id_ece, width, label='ID ECE')
        ax4.bar(x + width/2, ood_ece, width, label='OOD ECE')
        ax4.set_xlabel('Method')
        ax4.set_ylabel('ECE')
        ax4.set_title('Calibration Error (Lower is Better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45)
        ax4.legend()
        
        # 5. Accuracy Comparison
        ax5 = plt.subplot(3, 3, 5)
        id_acc = [results_dict[m]['metrics']['accuracy'] for m in methods]
        ood_acc = [results_dict[m]['ood_metrics']['accuracy'] for m in methods]
        ax5.bar(x - width/2, id_acc, width, label='ID Accuracy')
        ax5.bar(x + width/2, ood_acc, width, label='OOD Accuracy')
        ax5.set_xlabel('Method')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Classification Accuracy')
        ax5.set_xticks(x)
        ax5.set_xticklabels(methods, rotation=45)
        ax5.legend()
        
        # 6. σ_min vs OOD Performance
        ax6 = plt.subplot(3, 3, 6)
        sigma_mins = [results_dict[m]['metrics']['sigma_min'] for m in methods]
        ood_accs = [results_dict[m]['ood_metrics']['accuracy'] for m in methods]
        scatter = ax6.scatter(sigma_mins, ood_accs, s=100)
        for i, method in enumerate(methods):
            ax6.annotate(method, (sigma_mins[i], ood_accs[i]), fontsize=8)
        ax6.set_xlabel('σ_min')
        ax6.set_ylabel('OOD Accuracy')
        ax6.set_title('σ_min vs OOD Performance')
        
        # Add correlation line
        z = np.polyfit(sigma_mins, ood_accs, 1)
        p = np.poly1d(z)
        ax6.plot(sigma_mins, p(sigma_mins), "r--", alpha=0.8)
        
        # 7. Feature Distribution (t-SNE)
        ax7 = plt.subplot(3, 3, 7)
        # Use 'none' as baseline
        baseline_features = results_dict['none']['features'][:500].cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        baseline_2d = tsne.fit_transform(baseline_features)
        ax7.scatter(baseline_2d[:, 0], baseline_2d[:, 1], alpha=0.5, s=10)
        ax7.set_title('Baseline Feature Distribution (t-SNE)')
        
        # 8. Feature Distribution (Best Method)
        ax8 = plt.subplot(3, 3, 8)
        best_method = max(methods, key=lambda m: results_dict[m]['ood_metrics']['accuracy'])
        best_features = results_dict[best_method]['features'][:500].cpu().numpy()
        best_2d = tsne.fit_transform(best_features)
        ax8.scatter(best_2d[:, 0], best_2d[:, 1], alpha=0.5, s=10)
        ax8.set_title(f'{best_method} Feature Distribution (t-SNE)')
        
        # 9. Eigenvalue Distribution
        ax9 = plt.subplot(3, 3, 9)
        for name, data in results_dict.items():
            eigenvals = data['eigenvalues'].cpu().numpy()
            ax9.plot(np.sort(eigenvals)[::-1][:50], label=name, linewidth=2)
        ax9.set_xlabel('Index')
        ax9.set_ylabel('Eigenvalue')
        ax9.set_title('Top 50 Eigenvalues of Covariance Matrix')
        ax9.legend()
        ax9.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self, reg_strengths=None):
        """Run complete experiment comparing all regularization methods."""
        if reg_strengths is None:
            reg_strengths = {
                'none': 0.0,
                'geodesic': 0.1,
                'spectral': 0.01,
                'vmf': 0.1,
                'orthogonal': 1.0,
                'grassmann': 0.1
            }
        
        # Generate data
        print("Generating synthetic data...")
        data = self.generate_synthetic_data(n_samples=2000)
        
        # Split into train/test
        n_train = 1500
        train_data = {
            'id_features': data['id_features'][:n_train],
            'ood_features': data['ood_features'][:n_train],
            'text_features': data['text_features'],
            'labels': data['labels'][:n_train]
        }
        
        test_data = {
            'id_features': data['id_features'][n_train:],
            'ood_features': data['ood_features'][n_train:],
            'text_features': data['text_features'],
            'labels': data['labels'][n_train:]
        }
        
        results = {}
        
        # Test each regularization method
        for reg_type in tqdm(reg_strengths.keys(), desc="Testing regularizations"):
            print(f"\nTesting {reg_type} regularization...")
            
            # Apply regularization
            if reg_type == 'orthogonal':
                # Create a dummy projection matrix
                proj_matrix = nn.Parameter(torch.randn(self.feature_dim, self.feature_dim).to(self.device))
                regularized_features = self.apply_regularization(
                    train_data['id_features'], 
                    train_data['text_features'],
                    reg_type, 
                    reg_strengths[reg_type],
                    projection_matrix=proj_matrix
                )
            else:
                regularized_features = self.apply_regularization(
                    train_data['id_features'], 
                    train_data['text_features'],
                    reg_type, 
                    reg_strengths[reg_type],
                    pretrained_features=train_data['id_features'].clone()
                )
            
            # Compute metrics
            id_metrics, S, eigenvalues = self.compute_metrics(
                regularized_features, 
                train_data['labels'], 
                train_data['text_features']
            )
            
            # For OOD, we simulate the feature transformation
            ood_features_transformed = regularized_features + torch.randn_like(regularized_features) * 0.1
            ood_features_transformed = F.normalize(ood_features_transformed, p=2, dim=1)
            
            ood_metrics, _, _ = self.compute_metrics(
                ood_features_transformed,
                train_data['labels'],
                train_data['text_features']
            )
            
            results[reg_type] = {
                'features': regularized_features,
                'metrics': id_metrics,
                'ood_metrics': ood_metrics,
                'singular_values': S,
                'eigenvalues': eigenvalues
            }
            
            print(f"ID σ_min: {id_metrics['sigma_min']:.4f}, "
                  f"OOD Acc: {ood_metrics['accuracy']:.4f}, "
                  f"ECE: {ood_metrics['ece']:.4f}")
        
        return results
    
    def ablation_study(self, reg_type='spectral', strengths=None):
        """Study the effect of varying regularization strength."""
        if strengths is None:
            strengths = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
        
        # Generate data
        data = self.generate_synthetic_data(n_samples=1000)
        
        results = {
            'strengths': strengths,
            'sigma_mins': [],
            'ood_accs': [],
            'eces': [],
            'condition_numbers': []
        }
        
        for strength in tqdm(strengths, desc=f"Ablation for {reg_type}"):
            # Apply regularization
            regularized_features = self.apply_regularization(
                data['id_features'], 
                data['text_features'],
                reg_type, 
                strength,
                pretrained_features=data['id_features'].clone()
            )
            
            # Compute metrics
            metrics, _, _ = self.compute_metrics(
                regularized_features,
                data['labels'],
                data['text_features']
            )
            
            results['sigma_mins'].append(metrics['sigma_min'])
            results['ood_accs'].append(metrics['accuracy'])
            results['eces'].append(metrics['ece'])
            results['condition_numbers'].append(metrics['condition_number'])
        
        # Visualize ablation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].semilogx(strengths, results['sigma_mins'], 'o-')
        axes[0, 0].set_xlabel('Regularization Strength')
        axes[0, 0].set_ylabel('σ_min')
        axes[0, 0].set_title(f'{reg_type}: Effect on σ_min')
        axes[0, 0].grid(True)
        
        axes[0, 1].semilogx(strengths, results['ood_accs'], 'o-')
        axes[0, 1].set_xlabel('Regularization Strength')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title(f'{reg_type}: Effect on Accuracy')
        axes[0, 1].grid(True)
        
        axes[1, 0].semilogx(strengths, results['eces'], 'o-')
        axes[1, 0].set_xlabel('Regularization Strength')
        axes[1, 0].set_ylabel('ECE')
        axes[1, 0].set_title(f'{reg_type}: Effect on Calibration')
        axes[1, 0].grid(True)
        
        axes[1, 1].loglog(strengths, results['condition_numbers'], 'o-')
        axes[1, 1].set_xlabel('Regularization Strength')
        axes[1, 1].set_ylabel('Condition Number')
        axes[1, 1].set_title(f'{reg_type}: Effect on Conditioning')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'ablation_{reg_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results


class SyntheticCLIPExperiment:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.clip_model, self.train_preprocess, self.test_preprocess = clip.load("ViT-B/16", device=device, jit=False)
        self.clip_model.eval() # Keep CLIP in eval mode
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.feature_dim = self.clip_model.visual.output_dim # Usually 512 for ViT-B/16 image, 512 for text

    # --- 1. Enhanced Data Generation ---
    def generate_image_like_data_and_clip_features(self, n_samples_per_class=100, n_classes=5,
                                                    ood_shift_type='style', shift_magnitude=0.5, img_size=224):
        torch.manual_seed(42)
        np.random.seed(42)
        
        all_pseudo_images_id = []
        all_pseudo_images_ood = []
        all_labels = []
        
        # Define some basic class patterns (e.g., color patches)
        class_patterns = torch.rand(n_classes, 3, img_size // 8, img_size // 8) * 0.8 + 0.1 # Brighter patches

        for class_idx in range(n_classes):
            for _ in range(n_samples_per_class):
                # Base pseudo-image (e.g., noisy background)
                base_img = torch.rand(3, img_size, img_size) * 0.3 # Darker noise

                # Add class-specific pattern for ID
                id_img = base_img.clone()
                start_x, start_y = np.random.randint(0, img_size - class_patterns.shape[2], 2)
                id_img[:, start_x:start_x+class_patterns.shape[2], start_y:start_y+class_patterns.shape[3]] = class_patterns[class_idx]
                all_pseudo_images_id.append(torch.clamp(id_img, 0, 1))

                # Create OOD version
                ood_img = base_img.clone() # Start from same base for controlled shift
                ood_img[:, start_x:start_x+class_patterns.shape[2], start_y:start_y+class_patterns.shape[3]] = class_patterns[class_idx]


                if ood_shift_type == 'style':
                    # Example: Add significant Gaussian noise as style shift
                    noise = torch.randn_like(ood_img) * shift_magnitude * 0.5 # Scale magnitude
                    ood_img = torch.clamp(ood_img + noise, 0, 1)
                elif ood_shift_type == 'attribute':
                    # Example: Slightly change the pattern color
                    perturbed_pattern = torch.clamp(class_patterns[class_idx] + torch.rand_like(class_patterns[class_idx]) * shift_magnitude - shift_magnitude/2, 0, 1)
                    ood_img[:, start_x:start_x+class_patterns.shape[2], start_y:start_y+class_patterns.shape[3]] = perturbed_pattern
                # Add more sophisticated shifts if needed

                all_pseudo_images_ood.append(torch.clamp(ood_img, 0, 1))
                all_labels.append(class_idx)

        all_pseudo_images_id = torch.stack(all_pseudo_images_id).to(self.device)
        all_pseudo_images_ood = torch.stack(all_pseudo_images_ood).to(self.device)
        labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)

        # Get text features from CLIP
        class_prompts = [f"a synthetic image of class {chr(65+i)}" for i in range(n_classes)]
        tokenized_prompts = clip.tokenize(class_prompts).to(self.device)
        with torch.no_grad():
            class_text_features = self.clip_model.encode_text(tokenized_prompts)
            class_text_features = F.normalize(class_text_features, p=2, dim=1)

        # Get initial image features from CLIP visual encoder
        batch_size = 64 # Process in batches to avoid OOM with CLIP
        id_clip_image_features_list = []
        ood_clip_image_features_list = []

        with torch.no_grad():
            for i in range(0, len(all_pseudo_images_id), batch_size):
                id_batch = all_pseudo_images_id[i:i+batch_size]
                ood_batch = all_pseudo_images_ood[i:i+batch_size]
                # Preprocess for CLIP (minimal, as they are already tensors)
                # If using actual CLIP preprocess, it expects PIL Images
                # Here, we assume pseudo_images are [0,1] and directly pass to visual encoder
                # For real use, you'd use self.test_preprocess for images
                id_clip_image_features_list.append(self.clip_model.visual(id_batch.type(self.clip_model.dtype)))
                ood_clip_image_features_list.append(self.clip_model.visual(ood_batch.type(self.clip_model.dtype)))
        
        id_clip_image_features = F.normalize(torch.cat(id_clip_image_features_list), p=2, dim=1)
        ood_clip_image_features = F.normalize(torch.cat(ood_clip_image_features_list), p=2, dim=1)
        
        return {
            'id_clip_image_features': id_clip_image_features,      # From CLIP visual encoder
            'ood_clip_image_features': ood_clip_image_features,    # From CLIP visual encoder
            'class_text_features': class_text_features,            # From CLIP text encoder
            'labels': labels,
            'n_classes': n_classes
        }

    # --- 2. Trainable Model (Adapter/Head) ---
    class FeatureAdapter(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=1):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim # Should match text feature dim for contrastive loss
            
            if num_layers == 1:
                self.projection = nn.Linear(input_dim, output_dim)
            else:
                if hidden_dim is None:
                    hidden_dim = (input_dim + output_dim) // 2
                layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                for _ in range(num_layers - 2):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                layers.append(nn.Linear(hidden_dim, output_dim))
                self.projection = nn.Sequential(*layers)
            
            # For orthogonal regularization if applied to this layer's weight
            if num_layers == 1: # Only make sense for a single linear layer for direct W^T W - I
                 self.projection_weight_for_orth_reg = self.projection.weight # d_out x d_in
            else: # For MLP, orthogonality is less direct.
                 self.projection_weight_for_orth_reg = None


        def forward(self, x):
            return F.normalize(self.projection(x), p=2, dim=1)
        

    # --- 3. Fine-tuning/Regularization Loop ---
    def fine_tune_or_regularize_adapter(self, adapter_config, initial_image_features,
                                        class_text_features, labels, n_classes,
                                        reg_type='none', reg_strength=0.1,
                                        epochs=20, lr=1e-3, batch_size=64,
                                        clip_temp=0.07): # CLIP temperature

        adapter = self.FeatureAdapter(
            input_dim=self.feature_dim, 
            output_dim=class_text_features.shape[1], # Match text feature dim
            hidden_dim=adapter_config.get('hidden_dim'),
            num_layers=adapter_config.get('num_layers', 1)
        ).to(self.device)

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4 if reg_type == 'l2_decay' else 0)
        
        dataset = TensorDataset(initial_image_features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_img_features, batch_labels in loader:
                optimizer.zero_grad()
                
                adapted_img_features = adapter(batch_img_features) # N x D_adapter_out
                
                # Task Loss: CLIP Contrastive
                logits = torch.matmul(adapted_img_features, class_text_features.T) / clip_temp # N x N_classes_text
                
                # Create target labels for contrastive loss (image i matches text label[i])
                # This assumes class_text_features are ordered 0 to N_classes-1
                # and batch_labels correspond to these indices
                # For simplicity, let's assume classification-style cross-entropy on these logits for now
                # as the primary task. A true batch-wise contrastive is more complex here.
                # If we want true contrastive, text features should also be per-sample in batch.
                # For this synthetic setup, let's use image_features vs all_class_text_features.
                task_loss = F.cross_entropy(logits, batch_labels) # N_batch x N_classes vs N_batch
                
                reg_loss_val = torch.tensor(0.0).to(self.device)
                
                if reg_type == 'ortho_proj' and adapter.projection_weight_for_orth_reg is not None:
                    W = adapter.projection_weight_for_orth_reg # d_out x d_in
                    # We want W_proj^T W_proj - I if W_proj maps from input to output
                    # If W is (d_out, d_in), then W W^T is (d_out, d_out), W^T W is (d_in, d_in)
                    # CaRot applies to W_v (output of backbone -> proj_dim).
                    # Here adapter.projection.weight is D_adapter_out x D_adapter_in
                    # Let's assume we want the columns of W (mapping from each input dim) to be orthogonal,
                    # or rows (mapping to each output dim).
                    # If W is projection matrix itself (d_in -> d_out), then W^T W should be I (if d_out < d_in, approx)
                    # or W W^T should be I (if d_in < d_out, approx)
                    # CaRot's visual.proj is d_hidden -> d_embed. So W.T @ W.
                    # Here adapter.projection.weight is d_embed x d_feature_dim
                    gram_matrix = torch.matmul(adapter.projection_weight_for_orth_reg, adapter.projection_weight_for_orth_reg.T) # d_out x d_out
                    identity = torch.eye(gram_matrix.shape[0], device=self.device)
                    reg_loss_val = torch.norm(gram_matrix - identity, p='fro')**2
                
                elif reg_type == 'sigma_min_features':
                    if adapted_img_features.shape[0] > 1 and adapted_img_features.shape[1] > 1:
                         # svdvals needs at least 2x2. Also N > D for meaningful sigma_min of X
                        min_dim = min(adapted_img_features.shape)
                        if min_dim > 1 : # Make sure we can compute SVD
                            s = torch.linalg.svdvals(adapted_img_features)
                            reg_loss_val = -s.min() # Maximize smallest singular value
                        else:
                            reg_loss_val = torch.tensor(0.0).to(self.device)
                    else:
                        reg_loss_val = torch.tensor(0.0).to(self.device)

                elif reg_type == 'feature_covariance_logdet':
                    if adapted_img_features.shape[0] > adapted_img_features.shape[1]: # N > D for full rank cov
                        centered_features = adapted_img_features - adapted_img_features.mean(dim=0, keepdim=True)
                        cov_matrix = torch.matmul(centered_features.T, centered_features) / (centered_features.shape[0] -1)
                        # Add small epsilon for stability of logdet
                        cov_matrix += torch.eye(cov_matrix.shape[0], device=self.device) * 1e-6
                        # We want to maximize logdet(Cov), so minimize -logdet(Cov)
                        slogdet_val, slogdet_sign = torch.linalg.slogdet(cov_matrix)
                        if slogdet_sign > 0 : # ensure positive definite
                             reg_loss_val = -slogdet_val
                        else: # penalize if not PD
                             reg_loss_val = torch.tensor(10.0).to(self.device) # Large penalty
                    else:
                        reg_loss_val = torch.tensor(0.0).to(self.device) # Not well-defined otherwise


                total_loss = task_loss + reg_strength * reg_loss_val
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            if (epoch + 1) % (epochs // 5) == 0 or epochs < 5 :
                print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(loader):.4f}, Task: {task_loss.item():.4f}, Reg: {reg_loss_val.item():.4f}")
        
        return adapter
    
    # --- 4. Metrics Computation (largely similar, but on adapter outputs) ---
    def compute_metrics(self, output_features, labels, class_text_features_for_classifier, clip_temp=0.07):
        """Compute comprehensive metrics for evaluation on adapter's output features."""
        metrics = {}
        if output_features.numel() == 0 or output_features.shape[0] < 2 or output_features.shape[1] < 2:
            # Return default/nan metrics if features are too small
            return {
                'sigma_min_features': np.nan, 'sigma_max_features': np.nan, 'condition_number_features': np.nan,
                'effective_rank_features': np.nan, 'sigma_min_cov': np.nan, 'accuracy': np.nan,
                'ece': np.nan, 'mean_confidence': np.nan
            }, None, None

        # Singular value analysis of FEATURES (X)
        S_features = torch.linalg.svdvals(output_features)
        metrics['sigma_min_features'] = S_features.min().item() if S_features.numel() > 0 else np.nan
        metrics['sigma_max_features'] = S_features.max().item() if S_features.numel() > 0 else np.nan
        if S_features.numel() > 0 and S_features.min().item() > 1e-9:
            metrics['condition_number_features'] = (S_features.max() / S_features.min()).item()
        else:
            metrics['condition_number_features'] = np.inf
        
        S_normalized_feat = S_features / (S_features.sum() + 1e-10)
        entropy_feat = -(S_normalized_feat * torch.log(S_normalized_feat + 1e-10)).sum()
        metrics['effective_rank_features'] = torch.exp(entropy_feat).item()

        # Analysis of COVARIANCE MATRIX of features
        features_centered = output_features - output_features.mean(dim=0, keepdim=True)
        if features_centered.shape[0] <= features_centered.shape[1]: # N <= D
            # print(f"Warning: N ({features_centered.shape[0]}) <= D ({features_centered.shape[1]}). Covariance matrix might be rank-deficient.")
            # For N < D, X^T X is rank N.
            # We can compute SVD of X_centered and sigma_min(Cov) = sigma_min(X_centered)^2 / (N-1)
            # Or compute eigenvalues of X_centered @ X_centered.T (NxN) which are same as X_centered.T @ X_centered (DxD) non-zero ones
            # For simplicity, if N < D+1, sigma_min_cov might be zero or ill-defined.
            metrics['sigma_min_cov'] = 0.0 # Effectively
            eigenvalues_cov = torch.zeros(output_features.shape[1]).to(self.device) # Placeholder
        else:
            cov = torch.matmul(features_centered.t(), features_centered) / (output_features.size(0) - 1)
            try:
                eigenvalues_cov = torch.linalg.eigvalsh(cov) # For symmetric matrices
                metrics['sigma_min_cov'] = eigenvalues_cov.min().item() if eigenvalues_cov.numel() > 0 else np.nan
            except Exception as e:
                # print(f"Eigvalsh failed for covariance: {e}")
                metrics['sigma_min_cov'] = np.nan
                eigenvalues_cov = torch.zeros(output_features.shape[1]).to(self.device)

        # Calibration (ECE) and Accuracy
        # Logits for classification: output_features @ class_text_features.T
        similarities = torch.matmul(output_features, class_text_features_for_classifier.T) / clip_temp
        probs = F.softmax(similarities, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies_raw = (predictions == labels).float()
        
        metrics['accuracy'] = accuracies_raw.mean().item() if accuracies_raw.numel() > 0 else np.nan
        metrics['mean_confidence'] = confidences.mean().item() if confidences.numel() > 0 else np.nan
        
        ece = 0.0
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        
        if confidences.numel() > 0 :
            for i in range(n_bins):
                in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
                # Ensure the last bin includes 1.0
                if i == n_bins - 1:
                    in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])

                if in_bin.sum() > 0:
                    bin_confidence = confidences[in_bin].mean()
                    bin_accuracy = accuracies_raw[in_bin].mean()
                    ece += (in_bin.sum().float() / output_features.size(0)) * torch.abs(bin_confidence - bin_accuracy)
        metrics['ece'] = ece.item() if isinstance(ece, torch.Tensor) else ece # Ensure it's a float
        
        return metrics, S_features if S_features.numel() > 0 else None, eigenvalues_cov if eigenvalues_cov.numel() > 0 else None

    # --- 5. Visualization (largely similar) ---
    def visualize_results(self, results_dict, save_path='regularization_effects.png'):
        """Create comprehensive visualization of regularization effects."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Singular Value Spectrum
        ax1 = plt.subplot(3, 3, 1)
        for name, data in results_dict.items():
            S = data['singular_values_id_test']
            ax1.semilogy(S.cpu().numpy()[:100], label=name, linewidth=2)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value (log scale)')
        ax1.set_title('Singular Value Spectrum')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Condition Number Comparison
        ax2 = plt.subplot(3, 3, 2)
        methods = list(results_dict.keys())
        condition_numbers = [results_dict[m]['metrics_id_test']['condition_number_features'] for m in methods]
        bars = ax2.bar(methods, condition_numbers)
        ax2.set_ylabel('Condition Number')
        ax2.set_title('Condition Number (Lower is Better)')
        ax2.set_yscale('log')
        # Color bars based on value
        for bar, cn in zip(bars, condition_numbers):
            bar.set_color(plt.cm.RdYlGn_r(np.log10(cn) / 4))
        
        # 3. Effective Rank
        ax3 = plt.subplot(3, 3, 3)
        effective_ranks = [results_dict[m]['metrics_id_test']['effective_rank_features'] for m in methods]
        bars = ax3.bar(methods, effective_ranks)
        ax3.set_ylabel('Effective Rank')
        ax3.set_title('Effective Rank (Higher is Better)')
        for bar, er in zip(bars, effective_ranks):
            bar.set_color(plt.cm.RdYlGn(er / max(effective_ranks)))
        
        # 4. ECE Comparison
        ax4 = plt.subplot(3, 3, 4)
        id_ece = [results_dict[m]['metrics_id_test']['ece'] for m in methods]
        ood_ece = [results_dict[m]['metrics_ood_test']['ece'] for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        ax4.bar(x - width/2, id_ece, width, label='ID ECE')
        ax4.bar(x + width/2, ood_ece, width, label='OOD ECE')
        ax4.set_xlabel('Method')
        ax4.set_ylabel('ECE')
        ax4.set_title('Calibration Error (Lower is Better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45)
        ax4.legend()
        
        # 5. Accuracy Comparison
        ax5 = plt.subplot(3, 3, 5)
        id_acc = [results_dict[m]['metrics_id_test']['accuracy'] for m in methods]
        ood_acc = [results_dict[m]['metrics_ood_test']['accuracy'] for m in methods]
        ax5.bar(x - width/2, id_acc, width, label='ID Accuracy')
        ax5.bar(x + width/2, ood_acc, width, label='OOD Accuracy')
        ax5.set_xlabel('Method')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Classification Accuracy')
        ax5.set_xticks(x)
        ax5.set_xticklabels(methods, rotation=45)
        ax5.legend()
        
        # 6. σ_min vs OOD Performance
        ax6 = plt.subplot(3, 3, 6)
        sigma_mins = [results_dict[m]['metrics_id_test']['sigma_min_features'] for m in methods]
        ood_accs = [results_dict[m]['metrics_ood_test']['accuracy'] for m in methods]
        scatter = ax6.scatter(sigma_mins, ood_accs, s=100)
        for i, method in enumerate(methods):
            ax6.annotate(method, (sigma_mins[i], ood_accs[i]), fontsize=8)
        ax6.set_xlabel('σ_min')
        ax6.set_ylabel('OOD Accuracy')
        ax6.set_title('σ_min vs OOD Performance')
        
        # Add correlation line
        z = np.polyfit(sigma_mins, ood_accs, 1)
        p = np.poly1d(z)
        ax6.plot(sigma_mins, p(sigma_mins), "r--", alpha=0.8)
        
        # 7. Feature Distribution (t-SNE)
        ax7 = plt.subplot(3, 3, 7)
        # Use 'none' as baseline
        baseline_features = results_dict['none']['id_output_features_test'][:500].cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        baseline_2d = tsne.fit_transform(baseline_features)
        ax7.scatter(baseline_2d[:, 0], baseline_2d[:, 1], alpha=0.5, s=10)
        ax7.set_title('Baseline Feature Distribution (t-SNE)')
        
        # 8. Feature Distribution (Best Method)
        ax8 = plt.subplot(3, 3, 8)
        best_method = max(methods, key=lambda m: results_dict[m]['metrics_ood_test']['accuracy'])
        best_features = results_dict[best_method]['ood_output_features_test'][:500].cpu().numpy()
        best_2d = tsne.fit_transform(best_features)
        ax8.scatter(best_2d[:, 0], best_2d[:, 1], alpha=0.5, s=10)
        ax8.set_title(f'{best_method} Feature Distribution (t-SNE)')
        
        # 9. Eigenvalue Distribution
        ax9 = plt.subplot(3, 3, 9)
        for name, data in results_dict.items():
            eigenvals = data['eigenvalues_cov_id_test'].cpu().numpy()
            ax9.plot(np.sort(eigenvals)[::-1][:50], label=name, linewidth=2)
        ax9.set_xlabel('Index')
        ax9.set_ylabel('Eigenvalue')
        ax9.set_title('Top 50 Eigenvalues of Covariance Matrix')
        ax9.legend()
        ax9.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    # --- 6. Main Experiment Loop ---
    def run_experiment(self, reg_strengths_map=None, n_total_samples=1000, n_classes=5, 
                       train_split_ratio=0.7, ood_shift_type='style', shift_magnitude=0.3,
                       adapter_config={'num_layers': 1}, # Default to linear adapter
                       epochs_finetune=20, lr_finetune=1e-3):
        
        if reg_strengths_map is None:
            reg_strengths_map = {
                'none': 0.0,
                'l2_decay': 1e-4, # (Handled by AdamW weight_decay, reg_strength not used)
                'ortho_proj': 1.0,
                'sigma_min_features': 0.1,
                'feature_covariance_logdet': 0.01 
            }
        
        print("Generating image-like data and extracting initial CLIP features...")
        n_samples_per_class = n_total_samples // n_classes
        data = self.generate_image_like_data_and_clip_features(
            n_samples_per_class=n_samples_per_class, n_classes=n_classes, 
            ood_shift_type=ood_shift_type, shift_magnitude=shift_magnitude
        )
        
        # Split into train/test for CLIP features
        n_train = int(len(data['labels']) * train_split_ratio)
        indices = torch.randperm(len(data['labels']))
        train_indices, test_indices = indices[:n_train], indices[n_train:]

        train_id_clip_feats = data['id_clip_image_features'][train_indices]
        test_id_clip_feats = data['id_clip_image_features'][test_indices]
        test_ood_clip_feats = data['ood_clip_image_features'][test_indices] # Use original OOD features
        
        train_labels = data['labels'][train_indices]
        test_labels = data['labels'][test_indices] # Same labels for ID and OOD test
        
        class_text_features = data['class_text_features']
        
        results_log = {}
        
        for reg_type in tqdm(reg_strengths_map.keys(), desc="Testing regularizations"):
            print(f"\n--- Testing {reg_type} regularization (Strength: {reg_strengths_map.get(reg_type, 'N/A')}) ---")
            
            adapter = self.fine_tune_or_regularize_adapter(
                adapter_config=adapter_config,
                initial_image_features=train_id_clip_feats,
                class_text_features=class_text_features,
                labels=train_labels,
                n_classes=data['n_classes'],
                reg_type=reg_type,
                reg_strength=reg_strengths_map.get(reg_type, 0.0),
                epochs=epochs_finetune,
                lr=lr_finetune
            )
            adapter.eval() # Set adapter to eval mode for metric computation

            with torch.no_grad():
                # Get output features from the trained adapter
                id_output_features_test = adapter(test_id_clip_feats)
                ood_output_features_test = adapter(test_ood_clip_feats) # Pass OOD CLIP features through SAME adapter

            # Compute metrics on TEST split
            id_metrics, S_id, eig_cov_id = self.compute_metrics(
                id_output_features_test, test_labels, class_text_features
            )
            ood_metrics, S_ood, eig_cov_ood = self.compute_metrics(
                ood_output_features_test, test_labels, class_text_features # Same labels and text features
            )
            
            results_log[reg_type] = {
                'id_output_features_test': id_output_features_test, # For later visualization
                'ood_output_features_test': ood_output_features_test,
                'metrics_id_test': id_metrics,
                'metrics_ood_test': ood_metrics,
                'singular_values_id_test': S_id, # Singular values of ID output features X
                'eigenvalues_cov_id_test': eig_cov_id # Eigenvalues of Cov(ID output features)
            }
            
            print(f"  ID Test -> σ_min_cov: {id_metrics.get('sigma_min_cov', np.nan):.4f}, Acc: {id_metrics.get('accuracy', np.nan):.4f}, ECE: {id_metrics.get('ece', np.nan):.4f}")
            print(f"  OOD Test -> σ_min_cov: {ood_metrics.get('sigma_min_cov', np.nan):.4f}, Acc: {ood_metrics.get('accuracy', np.nan):.4f}, ECE: {ood_metrics.get('ece', np.nan):.4f}")

        return results_log
    
    # --- 7. Ablation Study (largely similar) ---
    def ablation_study(self, reg_type='spectral', strengths=None):
        """Study the effect of varying regularization strength."""
        if strengths is None:
            strengths = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
        
        # Generate data
        data = self.generate_synthetic_data(n_samples=1000)
        
        results = {
            'strengths': strengths,
            'sigma_mins': [],
            'ood_accs': [],
            'eces': [],
            'condition_numbers': []
        }
        
        for strength in tqdm(strengths, desc=f"Ablation for {reg_type}"):
            # Apply regularization
            regularized_features = self.apply_regularization(
                data['id_features'], 
                data['text_features'],
                reg_type, 
                strength,
                pretrained_features=data['id_features'].clone()
            )
            
            # Compute metrics
            metrics, _, _ = self.compute_metrics(
                regularized_features,
                data['labels'],
                data['text_features']
            )
            
            results['sigma_mins'].append(metrics['sigma_min'])
            results['ood_accs'].append(metrics['accuracy'])
            results['eces'].append(metrics['ece'])
            results['condition_numbers'].append(metrics['condition_number'])
        
        # Visualize ablation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].semilogx(strengths, results['sigma_mins'], 'o-')
        axes[0, 0].set_xlabel('Regularization Strength')
        axes[0, 0].set_ylabel('σ_min')
        axes[0, 0].set_title(f'{reg_type}: Effect on σ_min')
        axes[0, 0].grid(True)
        
        axes[0, 1].semilogx(strengths, results['ood_accs'], 'o-')
        axes[0, 1].set_xlabel('Regularization Strength')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title(f'{reg_type}: Effect on Accuracy')
        axes[0, 1].grid(True)
        
        axes[1, 0].semilogx(strengths, results['eces'], 'o-')
        axes[1, 0].set_xlabel('Regularization Strength')
        axes[1, 0].set_ylabel('ECE')
        axes[1, 0].set_title(f'{reg_type}: Effect on Calibration')
        axes[1, 0].grid(True)
        
        axes[1, 1].loglog(strengths, results['condition_numbers'], 'o-')
        axes[1, 1].set_xlabel('Regularization Strength')
        axes[1, 1].set_ylabel('Condition Number')
        axes[1, 1].set_title(f'{reg_type}: Effect on Conditioning')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'ablation_{reg_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results


def main():
    print("Initializing experiment...")
    experiment = SyntheticCLIPExperiment()
    
    adapter_config_linear = {'num_layers': 1}
    # adapter_config_mlp = {'num_layers': 2, 'hidden_dim': experiment.feature_dim // 2}

    print("\n" + "="*50)
    print("Running main comparison experiment with Linear Adapter...")
    print("="*50)
    results = experiment.run_experiment(
        adapter_config=adapter_config_linear,
        n_total_samples=2000, n_classes=5,
        epochs_finetune=30, lr_finetune=5e-4, # Adjust epochs/lr
        ood_shift_type='style', shift_magnitude=0.6 # Increase shift
    )
    experiment.visualize_results(results, save_path='regularization_effects_realistic.png') 
    # ... (Rest of main: ablation, summary table using new metric names) ...

    # Update Summary Table
    print("\n" + "="*50)
    print("Summary of Results (Test Set)")
    print("="*50)
    header = (f"{'Method':<25} {'ID σ_min_Cov':<15} {'ID Acc':<10} {'ID ECE':<10} "
              f"{'OOD σ_min_Cov':<15} {'OOD Acc':<10} {'OOD ECE':<10}")
    print(header)
    print("-" * len(header))
    
    for method, data_dict in results.items():
        id_m = data_dict['metrics_id_test']
        ood_m = data_dict['metrics_ood_test']
        print(f"{method:<25} "
              f"{id_m.get('sigma_min_cov', np.nan):<15.4f} "
              f"{id_m.get('accuracy', np.nan):<10.4f} "
              f"{id_m.get('ece', np.nan):<10.4f} "
              f"{ood_m.get('sigma_min_cov', np.nan):<15.4f} "
              f"{ood_m.get('accuracy', np.nan):<10.4f} "
              f"{ood_m.get('ece', np.nan):<10.4f}")

    # Statistical Analysis update
    print("\n" + "="*50)
    print("Statistical Analysis (using ID features' sigma_min_cov vs OOD Acc)")
    print("="*50)
    
    # Using ID output feature's sigma_min_cov vs OOD accuracy from test set
    sigma_min_cov_id_list = [results[m]['metrics_id_test'].get('sigma_min_cov', 0) for m in results.keys() if results[m]['metrics_id_test'].get('sigma_min_cov') is not np.nan]
    ood_acc_list = [results[m]['metrics_ood_test'].get('accuracy', 0) for m in results.keys() if results[m]['metrics_id_test'].get('sigma_min_cov') is not np.nan] # Ensure lists are same length

    if len(sigma_min_cov_id_list) > 1 and len(ood_acc_list) == len(sigma_min_cov_id_list):
        from scipy.stats import pearsonr
        valid_indices = [i for i, (s, a) in enumerate(zip(sigma_min_cov_id_list, ood_acc_list)) if not (np.isnan(s) or np.isnan(a))]
        if len(valid_indices) > 1:
            filtered_sigma_mins = [sigma_min_cov_id_list[i] for i in valid_indices]
            filtered_ood_accs = [ood_acc_list[i] for i in valid_indices]
            correlation, p_value = pearsonr(filtered_sigma_mins, filtered_ood_accs)
            print(f"Correlation between ID output features' σ_min_cov and OOD accuracy: {correlation:.4f} (p={p_value:.4f})")
        else:
            print("Not enough valid data points for correlation after filtering NaNs.")
    else:
        print("Not enough data points for correlation.")
    
    # (Feature space visualization would also use data from results_log[method]['id_output_features_test'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    methods_to_viz = ['none', 'l2_decay', 'ortho_proj', 'sigma_min_features', 'feature_covariance_logdet']

    for idx, method in enumerate(methods_to_viz):
        ax = axes[idx // 3, idx % 3]
        features = results[method]['id_output_features_test'][:500].cpu().numpy()
        
        # Project to 2D using PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create density plot
        x = features_2d[:, 0]
        y = features_2d[:, 1]
        
        # Calculate the point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        scatter = ax.scatter(x, y, c=z, s=10, alpha=0.5, cmap='viridis')
        ax.set_title(f'{method} (σ_min={results[method]["metrics_id_test"]["sigma_min_features"]:.4f})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()