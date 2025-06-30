import torch
import torch.nn.functional as F
import numpy as np
import wandb
from collections import defaultdict
import copy


class GradientDiagnostics:
    """
    Deep diagnostic system for tracking gradient components and their dynamics
    during CaRot training. Focuses on understanding natural dynamics without
    gradient modification.
    """
    
    def __init__(self, model, args, logger, log_frequency=50):
        """
        Initialize gradient diagnostics.
        
        Args:
            model: The CLIP model being trained
            args: Training arguments
            logger: Logger instance
            log_frequency: Log metrics every K steps
        """
        self.model = model
        self.args = args
        self.logger = logger
        self.log_frequency = log_frequency
        
        # Store pre-trained weights as reference
        self.pretrained_weights = {}
        self._store_pretrained_weights()
        
        # Key components to track (focusing on Wv, Wl initially)
        self.tracked_components = self._get_tracked_components()
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        
        # Training stage tracking
        self.total_steps = args.epochs * getattr(args, 'num_batches', 2503)  # Will be updated
        
    def _store_pretrained_weights(self):
        """Store pre-trained weights as reference points."""
        self.logger.info("Storing pre-trained weights for gradient diagnostics")
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.pretrained_weights[name] = param.data.clone().detach()
                
        self.logger.info(f"Stored {len(self.pretrained_weights)} pre-trained weight tensors")
    
    def _get_tracked_components(self):
        """Define key components to track (Wv, Wl, key backbone layers)."""
        tracked = {}
        
        # Focus on key projection layers first
        key_patterns = [
            'visual.proj',  # Wv - Vision projection
            'text_projection',  # Wl - Text projection  
            'visual.transformer.resblocks.11',  # Last vision transformer layer
            'transformer.resblocks.11',  # Last text transformer layer
            'visual.transformer.resblocks.6',  # Mid vision transformer layer
            'transformer.resblocks.6',  # Mid text transformer layer
        ]
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                for pattern in key_patterns:
                    if pattern in name:
                        tracked[name] = param
                        break
                        
        self.logger.info(f"Tracking gradients for {len(tracked)} key components:")
        for name in tracked.keys():
            self.logger.info(f"  - {name}")
            
        return tracked
    
    def compute_individual_gradients(self, losses, retain_graph=True):
        """
        Compute individual gradient components for each loss term.
        
        Args:
            losses: Dict containing individual loss terms
                   {'infonce': loss, 'oc': loss, 'crossf': loss, 'sd': loss}
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dict of gradient components per parameter
        """
        gradients = {}
        
        # Zero out any existing gradients
        self.model.zero_grad()
        
        # Compute gradients for each loss component
        for loss_name, loss_value in losses.items():
            if loss_value is None or loss_value == 0:
                continue
                
            # Compute gradients for this loss component
            component_grads = torch.autograd.grad(
                outputs=loss_value,
                inputs=self.tracked_components.values(),
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True
            )
            
            # Store gradients by parameter name
            for (param_name, param), grad in zip(self.tracked_components.items(), component_grads):
                if grad is not None:
                    if param_name not in gradients:
                        gradients[param_name] = {}
                    gradients[param_name][loss_name] = grad.detach().clone()
        
        return gradients
    
    def compute_reference_directions(self):
        """Compute direction vectors towards pre-trained weights."""
        directions = {}
        
        for name, current_param in self.tracked_components.items():
            if name in self.pretrained_weights:
                # d_pretrain = W_pretrain - W_current
                direction = self.pretrained_weights[name] - current_param.data
                directions[name] = direction
                
        return directions
    
    def compute_gradient_metrics(self, gradients, directions, step):
        """
        Compute comprehensive gradient metrics.
        
        Args:
            gradients: Individual gradient components
            directions: Reference directions
            step: Current training step
        """
        metrics = {}
        training_stage = self._get_training_stage(step)
        
        for param_name in self.tracked_components.keys():
            if param_name not in gradients:
                continue
                
            param_gradients = gradients[param_name]
            param_metrics = {}
            
            # 1. Gradient Norms
            for loss_name, grad in param_gradients.items():
                norm = torch.norm(grad).item()
                param_metrics[f"norm_{loss_name}"] = norm
                
            # 2. Reference direction norm
            if param_name in directions:
                d_pretrain = directions[param_name]
                param_metrics["norm_d_pretrain"] = torch.norm(d_pretrain).item()
                
                # 3. Cosine similarities with pretrained direction
                for loss_name, grad in param_gradients.items():
                    cos_sim = F.cosine_similarity(
                        grad.flatten(), d_pretrain.flatten(), dim=0
                    ).item()
                    param_metrics[f"cos_sim_{loss_name}_pretrain"] = cos_sim
            
            # 4. Pairwise cosine similarities between gradient components
            grad_names = list(param_gradients.keys())
            for i, loss_name1 in enumerate(grad_names):
                for loss_name2 in grad_names[i+1:]:
                    grad1 = param_gradients[loss_name1]
                    grad2 = param_gradients[loss_name2]
                    
                    cos_sim = F.cosine_similarity(
                        grad1.flatten(), grad2.flatten(), dim=0
                    ).item()
                    param_metrics[f"cos_sim_{loss_name1}_{loss_name2}"] = cos_sim
            
            # 5. Total gradient (sum of all components)
            if len(param_gradients) > 1:
                total_grad = sum(param_gradients.values())
                param_metrics["norm_total"] = torch.norm(total_grad).item()
                
                # Cosine similarity of total gradient with pretrained direction
                if param_name in directions:
                    cos_sim_total = F.cosine_similarity(
                        total_grad.flatten(), directions[param_name].flatten(), dim=0
                    ).item()
                    param_metrics["cos_sim_total_pretrain"] = cos_sim_total
            
            # Store metrics with proper naming
            for metric_name, value in param_metrics.items():
                full_name = f"{param_name.replace('.', '_')}_{metric_name}"
                metrics[full_name] = value
                
                # Also store by training stage
                stage_name = f"stage_{training_stage}_{full_name}"
                metrics[stage_name] = value
        
        return metrics
    
    def _get_training_stage(self, step):
        """Determine training stage: early, mid, late."""
        progress = step / self.total_steps
        if progress < 0.33:
            return "early"
        elif progress < 0.67:
            return "mid"
        else:
            return "late"
    
    def log_diagnostics(self, step, losses, performance_metrics=None):
        """
        Main logging function - call this during training.
        
        Args:
            step: Current training step
            losses: Dict of individual loss components
            performance_metrics: Optional dict of ID/OOD performance metrics
        """
        if step % self.log_frequency != 0:
            return
            
        self.logger.info(f"Computing gradient diagnostics at step {step}")
        
        try:
            # Compute individual gradients
            gradients = self.compute_individual_gradients(losses)
            
            # Compute reference directions
            directions = self.compute_reference_directions()
            
            # Compute all metrics
            metrics = self.compute_gradient_metrics(gradients, directions, step)
            
            # Add step and stage info
            metrics["diagnostics_step"] = step
            metrics["training_stage"] = self._get_training_stage(step)
            
            # Add performance metrics if provided
            if performance_metrics:
                for key, value in performance_metrics.items():
                    metrics[f"performance_{key}"] = value
            
            # Log to wandb with special prefix for gradient diagnostics
            wandb_metrics = {f"grad_diag/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)
            
            # Store in history for analysis
            self.metrics_history[step] = metrics
            
            # Log summary statistics
            self._log_summary_statistics(metrics, step)
            
        except Exception as e:
            self.logger.warning(f"Error in gradient diagnostics at step {step}: {e}")
    
    def _log_summary_statistics(self, metrics, step):
        """Log summary statistics about gradient dynamics."""
        
        # Find dominant gradient components by norm
        norm_metrics = {k: v for k, v in metrics.items() if 'norm_' in k and 'pretrain' not in k}
        
        if norm_metrics:
            max_norm_metric = max(norm_metrics.items(), key=lambda x: x[1])
            self.logger.info(f"Step {step} - Dominant gradient: {max_norm_metric[0]} = {max_norm_metric[1]:.6f}")
            
        # Check for concerning cosine similarities (potential conflicts)
        cos_sim_metrics = {k: v for k, v in metrics.items() if 'cos_sim_' in k and 'pretrain' not in k}
        negative_similarities = {k: v for k, v in cos_sim_metrics.items() if v < -0.5}
        
        if negative_similarities:
            self.logger.info(f"Step {step} - Strong gradient conflicts detected:")
            for metric, value in negative_similarities.items():
                self.logger.info(f"  {metric}: {value:.3f}")
    
    def update_total_steps(self, total_steps):
        """Update total steps for accurate stage computation."""
        self.total_steps = total_steps
    
    def get_gradient_conflicts_summary(self):
        """Get summary of gradient conflicts across training."""
        conflicts = defaultdict(list)
        
        for step, metrics in self.metrics_history.items():
            for metric_name, value in metrics.items():
                if 'cos_sim_' in metric_name and 'pretrain' not in metric_name:
                    if value < -0.3:  # Threshold for conflict
                        conflicts[metric_name].append((step, value))
        
        return dict(conflicts)


def create_loss_dict_for_diagnostics(ft_clip_loss, args, **loss_components):
    """
    Helper function to create properly separated loss components for diagnostics.
    
    Args:
        ft_clip_loss: Base InfoNCE loss
        args: Training arguments
        **loss_components: Individual loss components (oc_loss, crossf_loss, etc.)
    
    Returns:
        Dict of individual loss terms for gradient analysis
    """
    losses = {
        'infonce': ft_clip_loss,
    }
    
    # Add orthogonality constraint if present
    if args.l_orth_wv > 0 and 'oc_loss' in loss_components:
        losses['oc'] = args.l_orth_wv * loss_components['oc_loss']
    
    # Add cross F-norm if present
    if args.cross_fnorm > 0 and 'crossf_loss' in loss_components:
        losses['crossf'] = args.cross_fnorm * loss_components['crossf_loss']
    
    # Add self-distillation if present
    if args.distil_coef > 0 and 'sd_loss' in loss_components:
        losses['sd'] = args.distil_coef * loss_components['sd_loss']
    
    return losses 