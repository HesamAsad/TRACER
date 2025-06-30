#!/usr/bin/env python3
"""
Test script for gradient diagnostics functionality.
Demonstrates the deep diagnostic system for CaRot training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import logging
import wandb

# Mock classes for testing
class MockCLIPModel(nn.Module):
    """Mock CLIP model for testing gradient diagnostics."""
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Vision projection (Wv)
        self.visual = nn.Module()
        self.visual.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Text projection (Wl) 
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # Mock transformer layers
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(12)
        ])
        
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(12)
        ])
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, images, texts):
        # Simplified forward pass
        batch_size = images.shape[0]
        device = images.device
        
        # Mock image and text features (ensure they're on the same device)
        image_features = torch.randn(batch_size, self.embed_dim, device=device)
        text_features = torch.randn(batch_size, self.embed_dim, device=device)
        
        # Apply projections
        image_features = image_features @ self.visual.proj.weight.T
        text_features = text_features @ self.text_projection
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features, self.logit_scale

class MockArgs:
    """Mock arguments for testing."""
    def __init__(self):
        self.l_orth_wv = 0.01
        self.cross_fnorm = 0.005
        self.distil_coef = 0.1
        self.enable_grad_diagnostics = True
        self.epochs = 2
        self.model = "ViT-B/32"

def mock_clip_loss_fn(image_features, text_features, logit_scale):
    """Mock CLIP loss function."""
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2
    
    return loss, logits_per_image, logits_per_text

def compute_mock_losses(model, image_features, text_features, logit_scale):
    """Compute mock loss components for testing."""
    
    # Base InfoNCE loss
    infonce_loss, _, _ = mock_clip_loss_fn(image_features, text_features, logit_scale)
    
    # Orthogonality constraint loss
    covv = model.visual.proj.weight.T @ model.visual.proj.weight
    oc_loss = ((covv - torch.eye(covv.shape[0], device=covv.device))**2).sum()**(1/2)
    
    # Cross F-norm loss
    cov_vl = model.visual.proj.weight @ model.text_projection.T
    crossf_loss = torch.linalg.norm(cov_vl, ord='fro')
    
    # Mock self-distillation loss (ensure it's on the same device)
    device = image_features.device
    sd_loss = 0.1 * torch.randn(1, requires_grad=True, device=device).squeeze()
    
    return {
        'infonce': infonce_loss,
        'oc': oc_loss,
        'crossf': crossf_loss,
        'sd': sd_loss
    }

def test_gradient_diagnostics():
    """Test the gradient diagnostics system."""
    
    # Setup
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize mock model
    model = MockCLIPModel().to(device)
    args = MockArgs()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Import gradient diagnostics
    try:
        from src.models.gradient_diagnostics import GradientDiagnostics, create_loss_dict_for_diagnostics
    except ImportError:
        print("Could not import gradient diagnostics module. Make sure it's in the correct path.")
        return
    
    # Initialize gradient diagnostics
    grad_diagnostics = GradientDiagnostics(
        model=model,
        args=args,
        logger=logger,
        log_frequency=5
    )
    
    # Mock data
    batch_size = 32
    embed_dim = 512
    
    # Store metrics for visualization
    metrics_history = defaultdict(list)
    
    print("Starting gradient diagnostics test...")
    
    # Simulate training steps
    num_steps = 50
    for step in range(num_steps):
        
        # Generate mock batch
        images = torch.randn(batch_size, embed_dim).to(device)
        texts = torch.randn(batch_size, embed_dim).to(device)
        
        # Forward pass
        image_features, text_features, logit_scale = model(images, texts)
        
        # Compute individual losses
        mock_losses = compute_mock_losses(model, image_features, text_features, logit_scale)
        
        # Create loss dict for diagnostics
        loss_dict = create_loss_dict_for_diagnostics(
            mock_losses['infonce'], args,
            oc_loss=mock_losses['oc'],
            crossf_loss=mock_losses['crossf'],
            sd_loss=mock_losses['sd']
        )
        
        # Mock performance metrics
        performance_metrics = {
            'id_acc': 0.7 + 0.1 * np.sin(step * 0.1),  # Oscillating accuracy
            'ood_acc': 0.5 + 0.05 * np.cos(step * 0.15),
            'total_loss': 2.0 - step * 0.02,  # Decreasing loss
        }
        
        # Run diagnostics
        grad_diagnostics.log_diagnostics(step, loss_dict, performance_metrics)
        
        # Compute total loss and backward pass (to update weights)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        
        # Simple SGD step
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= 0.01 * param.grad
                    param.grad.zero_()
        
        # Store some metrics for visualization
        if step % grad_diagnostics.log_frequency == 0:
            print(f"Step {step}: Total loss = {total_loss.item():.4f}")
            
            # Extract some key metrics from diagnostics history
            if step in grad_diagnostics.metrics_history:
                step_metrics = grad_diagnostics.metrics_history[step]
                for key, value in step_metrics.items():
                    if 'norm_' in key:
                        metrics_history[key].append((step, value))
    
    print("Gradient diagnostics test completed!")
    
    # Generate conflict summary
    conflicts = grad_diagnostics.get_gradient_conflicts_summary()
    print("\nGradient Conflicts Summary:")
    if conflicts:
        for conflict_name, conflict_data in conflicts.items():
            print(f"  {conflict_name}: {len(conflict_data)} instances")
            if conflict_data:
                min_val = min(conflict_data, key=lambda x: x[1])
                max_val = max(conflict_data, key=lambda x: x[1])
                print(f"    Range: {min_val[1]:.3f} to {max_val[1]:.3f}")
    else:
        print("  No significant conflicts detected")
    
    # Visualization
    visualize_gradient_metrics(metrics_history)
    
    return grad_diagnostics

def visualize_gradient_metrics(metrics_history):
    """Visualize gradient metrics over time."""
    
    if not metrics_history:
        print("No metrics to visualize")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot gradient norms
    plt.subplot(2, 2, 1)
    for metric_name, data in metrics_history.items():
        if 'norm_' in metric_name and 'pretrain' not in metric_name:
            steps, values = zip(*data)
            plt.plot(steps, values, label=metric_name.replace('_', ' ').title(), marker='o', markersize=3)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Component Norms Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot reference direction norms
    plt.subplot(2, 2, 2)
    for metric_name, data in metrics_history.items():
        if 'norm_d_pretrain' in metric_name:
            steps, values = zip(*data)
            component_name = metric_name.split('_norm_d_pretrain')[0]
            plt.plot(steps, values, label=f'{component_name} Distance to Pretrained', marker='s', markersize=3)
    plt.xlabel('Training Step')
    plt.ylabel('Distance Norm')
    plt.title('Distance from Pre-trained Weights')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot cosine similarities with pretrained
    plt.subplot(2, 2, 3)
    cosine_pretrain_metrics = {}
    for metric_name, data in metrics_history.items():
        if 'cos_sim_' in metric_name and '_pretrain' in metric_name:
            cosine_pretrain_metrics[metric_name] = data
    
    for metric_name, data in cosine_pretrain_metrics.items():
        steps, values = zip(*data)
        label = metric_name.replace('_', ' ').replace('cos sim', 'CosSim').title()
        plt.plot(steps, values, label=label, marker='^', markersize=3)
    plt.xlabel('Training Step')
    plt.ylabel('Cosine Similarity')
    plt.title('Gradient Components vs Pre-trained Direction')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Plot pairwise cosine similarities
    plt.subplot(2, 2, 4)
    pairwise_metrics = {}
    for metric_name, data in metrics_history.items():
        if 'cos_sim_' in metric_name and '_pretrain' not in metric_name:
            pairwise_metrics[metric_name] = data
    
    for metric_name, data in pairwise_metrics.items():
        steps, values = zip(*data)
        label = metric_name.replace('_', ' ').replace('cos sim', 'CosSim').title()
        plt.plot(steps, values, label=label, marker='d', markersize=3)
    plt.xlabel('Training Step')
    plt.ylabel('Cosine Similarity')
    plt.title('Pairwise Gradient Component Similarities')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.5, color='r', linestyle=':', alpha=0.5, label='Conflict Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_diagnostics_test.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'gradient_diagnostics_test.png'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gradient diagnostics")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    
    # Initialize wandb in offline mode for testing
    if not args.no_wandb:
        wandb.init(project="gradient-diagnostics-test", mode="offline")
    
    # Run test
    diagnostics = test_gradient_diagnostics()
    
    if not args.no_wandb:
        wandb.finish()
    
    print("Test completed successfully!") 