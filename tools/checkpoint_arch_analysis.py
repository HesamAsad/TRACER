#!/usr/bin/env python3
"""
Comprehensive checkpoint and architecture analysis tool for CLIP models.

This module provides tools to compare Pretrained, Direct-FT (FLYP), and POMP-FT checkpoints
through parameter-delta analysis, representation-delta analysis (CKA/SVCCA), linear probes,
and weight-space interpolation (WiSE-style).
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import OrderedDict
import copy
import hashlib
import subprocess
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.modeling import CLIPEncoder
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from types import SimpleNamespace
import clip.clip as clip
# Note: Using OpenAI CLIP from clip/ folder, not open_clip
import src.datasets_ as datasets


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for checkpoint analysis."""
    pretrained_from_repo: bool = True
    ckpt_direct: Optional[str] = None
    ckpt_pomp: Optional[str] = None
    output_dir: str = "analysis"
    num_images_per_split: int = 10000
    batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda:0"
    seed: int = 42
    cache_features: bool = True
    datasets: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {}


def build_min_args(model: str,
                   device: str,
                   dataset: str = "ImageNet",
                   template: str = "openai_imagenet_template",
                   batch_size: int = 128,
                   workers: int = 8,
                   data_location: str = None,
                   use_fp16: int = 1,
                   seed: int = 42) -> SimpleNamespace:
    """Build minimal args namespace matching compare_models_and_gradcam.py pattern."""
    return SimpleNamespace(
        model=model,
        device=device,
        train_dataset=dataset,
        template=template,
        batch_size=batch_size,
        workers=workers,
        data_location=data_location or os.path.expanduser('~/data'),
        use_fp16=use_fp16,
        seed=seed,
        classnames="openai"
    )


class ModelLoader:
    """Utilities for loading pretrained and finetuned CLIP models."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def build_pretrained_model(self, model_name: str = "ViT-B/16") -> CLIPEncoder:
        """Build a fresh pretrained CLIP model matching the repo's canonical way."""
        logger.info(f"Building pretrained model: {model_name} using OpenAI CLIP (from clip/ folder)")
        
        # Use 'ViT-B/16' format (with slash) to ensure CLIPEncoder uses OpenAI CLIP via clip.load()
        # CLIPEncoder only uses open_clip for 'ViT-B-16' (hyphen) or 'ViT-L-14'
        # So 'ViT-B/16' (slash) will use the else branch: clip.load() → OpenAI CLIP
        model_name_for_clip = model_name  # Keep 'ViT-B/16' format
        
        # Build args matching compare_models_and_gradcam.py pattern
        args = build_min_args(
            model=model_name_for_clip,
            device=self.device,
            dataset="ImageNet",
            template="openai_imagenet_template"
        )
        
        # Build model - CLIPEncoder will use clip.load() for 'ViT-B/16' (OpenAI CLIP)
        clip_encoder = CLIPEncoder(args, keep_lang=True)
        clip_encoder = clip_encoder.to(self.device)
        clip_encoder.eval()
        return clip_encoder
    
    def load_checkpoint(self, checkpoint_path: str, model_name: str = "ViT-B/16") -> CLIPEncoder:
        """Load a finetuned checkpoint into a CLIPEncoder (matching compare_models_and_gradcam.py)."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Use 'ViT-B/16' format (with slash) to ensure OpenAI CLIP is used
        # When loading checkpoints, we still need to instantiate CLIPEncoder with correct args
        # to match the architecture. Using 'ViT-B/16' ensures OpenAI CLIP via clip.load()
        model_name_for_clip = model_name  # Keep 'ViT-B/16' format
        
        # Build args (matching compare_models_and_gradcam.py pattern)
        args = build_min_args(
            model=model_name_for_clip,
            device=self.device,
            dataset="ImageNet",
            template="openai_imagenet_template"
        )
        
        # Mirror compare_models_and_gradcam.py: instantiate then load via instance method
        # Note: The instantiation creates the right architecture, then load() replaces with checkpoint weights
        enc = CLIPEncoder(args, keep_lang=True)
        enc = enc.load(checkpoint_path)
        enc = enc.to(self.device)
        enc.eval()
        
        # Handle DataParallel prefix if present
        if hasattr(enc, 'model') and hasattr(enc.model, 'module'):
            logger.info("Removing DataParallel wrapper")
            enc.model = enc.model.module
        
        return enc
    
    def extract_state_dict(self, model: CLIPEncoder, strip_prefix: bool = True) -> OrderedDict:
        """Extract state dict from model, handling various formats (matching repo patterns)."""
        state_dict = model.state_dict()
        
        if strip_prefix:
            # Remove common prefixes (matching compare_models_and_gradcam.py patterns)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # Remove 'module.' prefix (DataParallel)
                if k.startswith('module.'):
                    k = k[7:]
                # Keep 'model.' prefix as CLIPEncoder wraps the underlying CLIP model
                # The state dict keys should match the underlying model structure
                new_state_dict[k] = v
            return new_state_dict
        
        return state_dict


class LayerMapper:
    """Extract and map layer structure from CLIP models."""
    
    def __init__(self, model: CLIPEncoder):
        self.model = model
        self.layer_map = self._build_layer_map()
    
    def _build_layer_map(self) -> Dict[str, Any]:
        """Build a canonical layer map for the model."""
        layer_map = {
            'vision': {
                'embeddings': [],
                'blocks': [],
                'projection': None,
            },
            'text': {
                'embeddings': [],
                'blocks': [],
                'projection': None,
            },
            'global': {
                'logit_scale': None,
            }
        }
        
        state_dict = self.model.state_dict()
        
        # Process vision encoder
        vision_keys = [k for k in state_dict.keys() if k.startswith('model.visual.') or 
                      (not k.startswith('model.transformer.') and 'visual' in k.lower())]
        
        for key in sorted(vision_keys):
            clean_key = key.replace('model.visual.', '').replace('module.', '')
            
            if 'conv1' in clean_key or 'class_embedding' in clean_key or 'positional_embedding' in clean_key:
                layer_map['vision']['embeddings'].append(clean_key)
            elif 'transformer.resblocks' in clean_key:
                # Extract block index
                parts = clean_key.split('.')
                block_idx = None
                for i, part in enumerate(parts):
                    if part == 'resblocks':
                        if i + 1 < len(parts):
                            try:
                                block_idx = int(parts[i + 1])
                                break
                            except ValueError:
                                pass
                
                if block_idx is not None:
                    # Ensure block list is long enough
                    while len(layer_map['vision']['blocks']) <= block_idx:
                        layer_map['vision']['blocks'].append({
                            'attn': {'qkv': None, 'proj': None},
                            'mlp': {'fc1': None, 'fc2': None},
                            'ln1': None,
                            'ln2': None,
                        })
                    
                    if 'attn.in_proj' in clean_key or 'attn.qkv' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['attn']['qkv'] = clean_key
                    elif 'attn.out_proj' in clean_key or 'attn.proj' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['attn']['proj'] = clean_key
                    elif 'mlp.c_fc' in clean_key or 'mlp.fc1' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['mlp']['fc1'] = clean_key
                    elif 'mlp.c_proj' in clean_key or 'mlp.fc2' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['mlp']['fc2'] = clean_key
                    elif 'ln_1' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['ln1'] = clean_key
                    elif 'ln_2' in clean_key:
                        layer_map['vision']['blocks'][block_idx]['ln2'] = clean_key
            elif 'proj' in clean_key and 'transformer' not in clean_key:
                layer_map['vision']['projection'] = clean_key
        
        # Process text encoder (if present)
        text_keys = [k for k in state_dict.keys() if 'transformer' in k and 'visual' not in k]
        # Similar processing for text...
        
        # Process global parameters
        if 'logit_scale' in state_dict:
            layer_map['global']['logit_scale'] = 'logit_scale'
        
        return layer_map
    
    def save_layer_map(self, output_path: str):
        """Save layer map to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.layer_map, f, indent=2)
        logger.info(f"Saved layer map to {output_path}")


class ParameterAnalyzer:
    """Analyze parameter differences between models."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def compute_parameter_metrics(
        self, 
        state_dict_1: OrderedDict, 
        state_dict_2: OrderedDict,
        name_1: str = "model_1",
        name_2: str = "model_2"
    ) -> pd.DataFrame:
        """Compute parameter-delta metrics between two state dicts."""
        results = []
        
        # Find common keys
        keys_1 = set(state_dict_1.keys())
        keys_2 = set(state_dict_2.keys())
        common_keys = keys_1 & keys_2
        only_1 = keys_1 - keys_2
        only_2 = keys_2 - keys_1
        
        logger.info(f"Common keys: {len(common_keys)}, Only in {name_1}: {len(only_1)}, Only in {name_2}: {len(only_2)}")
        
        for key in tqdm(sorted(common_keys), desc="Computing parameter metrics"):
            w1 = state_dict_1[key].float().cpu()
            w2 = state_dict_2[key].float().cpu()
            
            if w1.shape != w2.shape:
                logger.warning(f"Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
                continue
            
            # Flatten for some metrics
            w1_flat = w1.flatten()
            w2_flat = w2.flatten()
            delta = w2_flat - w1_flat
            
            # Compute metrics
            l2_norm_delta = torch.norm(delta, p=2).item()
            l2_norm_w1 = torch.norm(w1_flat, p=2).item()
            relative_delta = l2_norm_delta / (l2_norm_w1 + 1e-10)
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()
            
            # Frobenius norm (for 2D+ tensors)
            if w1.ndim >= 2:
                frobenius_delta = torch.norm(delta.reshape(w1.shape), p='fro').item()
                frobenius_w1 = torch.norm(w1, p='fro').item()
                relative_frobenius = frobenius_delta / (frobenius_w1 + 1e-10)
            else:
                frobenius_delta = l2_norm_delta
                frobenius_w1 = l2_norm_w1
                relative_frobenius = relative_delta
            
            # Spectral drift (top-k singular values)
            spectral_drift = self._compute_spectral_drift(w1, w2, k=5)
            
            # Parse layer info
            layer_info = self._parse_layer_info(key)
            
            results.append({
                'key': key,
                'layer_type': layer_info['type'],
                'block_idx': layer_info['block_idx'],
                'submodule': layer_info['submodule'],
                'l2_norm_delta': l2_norm_delta,
                'relative_delta': relative_delta,
                'cosine_similarity': cosine_sim,
                'frobenius_delta': frobenius_delta,
                'relative_frobenius': relative_frobenius,
                'spectral_drift': spectral_drift,
                'shape': str(list(w1.shape)),
            })
        
        df = pd.DataFrame(results)
        return df
    
    def _compute_spectral_drift(self, w1: torch.Tensor, w2: torch.Tensor, k: int = 5) -> float:
        """Compute spectral drift as mean absolute difference of top-k singular values."""
        # Reshape to 2D if needed
        if w1.ndim == 1:
            w1 = w1.unsqueeze(0)
            w2 = w2.unsqueeze(0)
        elif w1.ndim > 2:
            w1 = w1.reshape(w1.shape[0], -1)
            w2 = w2.reshape(w2.shape[0], -1)
        
        try:
            s1 = torch.linalg.svdvals(w1)[:k]
            s2 = torch.linalg.svdvals(w2)[:k]
            drift = torch.abs(s1 - s2).mean().item()
            return drift
        except Exception as e:
            logger.warning(f"Spectral drift computation failed: {e}")
            return 0.0
    
    def _parse_layer_info(self, key: str) -> Dict[str, Any]:
        """Parse layer information from state dict key."""
        info = {
            'type': 'unknown',
            'block_idx': None,
            'submodule': None,
        }
        
        if 'visual' in key:
            info['type'] = 'vision'
            if 'resblocks' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'resblocks' and i + 1 < len(parts):
                        try:
                            info['block_idx'] = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
                
                if 'attn' in key:
                    if 'in_proj' in key or 'qkv' in key:
                        info['submodule'] = 'attn.qkv'
                    elif 'out_proj' in key or 'proj' in key:
                        info['submodule'] = 'attn.proj'
                elif 'mlp' in key:
                    if 'c_fc' in key or 'fc1' in key:
                        info['submodule'] = 'mlp.fc1'
                    elif 'c_proj' in key or 'fc2' in key:
                        info['submodule'] = 'mlp.fc2'
                elif 'ln_1' in key:
                    info['submodule'] = 'ln1'
                elif 'ln_2' in key:
                    info['submodule'] = 'ln2'
            elif 'conv1' in key or 'class_embedding' in key or 'positional_embedding' in key:
                info['submodule'] = 'embedding'
            elif 'proj' in key:
                info['submodule'] = 'proj_head'
        elif 'transformer' in key and 'visual' not in key:
            info['type'] = 'text'
            # Extract block index for text transformer blocks
            if 'resblocks' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'resblocks' and i + 1 < len(parts):
                        try:
                            info['block_idx'] = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
                
                if 'attn' in key:
                    if 'in_proj' in key or 'qkv' in key:
                        info['submodule'] = 'attn.qkv'
                    elif 'out_proj' in key or 'proj' in key:
                        info['submodule'] = 'attn.proj'
                elif 'mlp' in key:
                    if 'c_fc' in key or 'fc1' in key:
                        info['submodule'] = 'mlp.fc1'
                    elif 'c_proj' in key or 'fc2' in key:
                        info['submodule'] = 'mlp.fc2'
                elif 'ln_1' in key:
                    info['submodule'] = 'ln1'
                elif 'ln_2' in key:
                    info['submodule'] = 'ln2'
        elif ('ln_final' in key or 'positional_embedding' in key or 
              'text_projection' in key or 'token_embedding' in key) and 'visual' not in key:
            info['type'] = 'text'
            if 'ln_final' in key:
                info['submodule'] = 'ln_final'
            elif 'positional_embedding' in key:
                info['submodule'] = 'positional_embedding'
            elif 'text_projection' in key:
                info['submodule'] = 'text_projection'
            elif 'token_embedding' in key:
                info['submodule'] = 'token_embedding'
        elif 'logit_scale' in key:
            info['type'] = 'global'
            info['submodule'] = 'logit_scale'
        
        return info


class RepresentationAnalyzer:
    """Analyze representation differences using CKA, SVCCA, and linear probes."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.feature_cache = {}
    
    def get_vision_hook_locations(self, model: CLIPEncoder) -> List[str]:
        """Get hook locations for vision transformer layers."""
        hook_locations = []
        
        # Get the visual transformer
        visual = model.model.visual
        
        # Check if it's a VisualTransformer (ViT) or ModifiedResNet
        if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            # ViT architecture
            num_layers = len(visual.transformer.resblocks)
            
            # Patch embedding (after conv1)
            hook_locations.append('patch_embed')
            
            # Exclude ln_pre - skip it
            
            # Each transformer block output
            for i in range(num_layers):
                hook_locations.append(f'block_{i}')
            
            # Post-norm (before projection)
            hook_locations.append('ln_post')
            
            # Final projection (after proj)
            hook_locations.append('final_proj')
        
        return hook_locations
    
    def extract_features(
        self,
        model: CLIPEncoder,
        dataloader: torch.utils.data.DataLoader,
        hook_locations: Optional[List[str]] = None,
        model_name: str = "model",
        cache_path: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract features from model at specified hook locations.
        
        Args:
            model: CLIP encoder model
            dataloader: DataLoader for images
            hook_locations: List of layer names to hook (auto-detected if None)
            model_name: Name for logging
            cache_path: Path to save/load cached features
            use_cache: If True, load from cache if available; if False, always recompute
        """
        if use_cache and cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached features from {cache_path}")
            cached_features = np.load(cache_path, allow_pickle=True).item()
            # Verify all hook locations are present
            if hook_locations:
                missing_locs = [loc for loc in hook_locations if loc not in cached_features]
                if missing_locs:
                    logger.warning(f"Cache missing locations {missing_locs}, recomputing...")
                else:
                    return cached_features
            else:
                return cached_features
        
        # Auto-detect hook locations if not provided
        if hook_locations is None:
            hook_locations = self.get_vision_hook_locations(model)
        
        model.eval()
        features = {loc: [] for loc in hook_locations}
        hooks = []
        
        visual = model.model.visual
        
        # Check architecture type
        is_vit = hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks')
        
        if not is_vit:
            raise ValueError("Currently only supports ViT architecture")
        
        # Register hooks for ViT
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                
                # Handle different output shapes
                if output.ndim == 3:  # [seq_len, batch, dim] or [batch, seq_len, dim]
                    # Check if it's LND (seq first) or NLD (batch first)
                    if output.shape[0] > output.shape[1]:  # Likely LND
                        output = output.permute(1, 0, 2)  # Convert to NLD
                    output = output[:, 0, :]  # Extract CLS token [batch, dim]
                elif output.ndim == 2:  # [batch, dim]
                    output = output
                else:
                    output = output.flatten(start_dim=1)
                
                features[name].append(output.detach().cpu())
            return hook_fn
        
        # Register hooks for each location
        hook_registry = {}
        
        # Patch embedding (after conv1 and reshape)
        def patch_embed_hook(module, input, output):
            # conv1 output: [batch, width, grid, grid]
            batch_size = output.shape[0]
            output_reshaped = output.reshape(batch_size, output.shape[1], -1)  # [batch, width, grid*grid]
            output_reshaped = output_reshaped.permute(0, 2, 1)  # [batch, grid*grid, width]
            # Take mean over patches (or could use CLS token position)
            output_mean = output_reshaped.mean(dim=1)  # [batch, width]
            features['patch_embed'].append(output_mean.detach().cpu())
        
        hook_registry['patch_embed'] = visual.conv1.register_forward_hook(patch_embed_hook)
        
        # Pre-norm - EXCLUDED per user request
        # hook_registry['ln_pre'] = visual.ln_pre.register_forward_hook(make_hook('ln_pre'))
        
        # Transformer blocks
        for i, block in enumerate(visual.transformer.resblocks):
            hook_registry[f'block_{i}'] = block.register_forward_hook(make_hook(f'block_{i}'))
        
        # Post-norm
        hook_registry['ln_post'] = visual.ln_post.register_forward_hook(make_hook('ln_post'))
        
        # Final projection (need to hook encode_image instead)
        # We'll extract this after the forward pass
        
        # Extract features
        final_features = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting features ({model_name})")):
                # Handle different batch formats
                # Standard ImageFolder returns (image, label) tuples
                # Some custom datasets return dicts with 'images' key
                if isinstance(batch, dict):
                    images = batch.get('images', batch.get('input', None))
                    if images is None:
                        # Try to get first value that's a tensor
                        images = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
                elif isinstance(batch, (list, tuple)):
                    # Could be (image, label) or just [image]
                    images = batch[0] if len(batch) > 0 else None
                elif isinstance(batch, torch.Tensor):
                    images = batch
                else:
                    logger.warning(f"Unexpected batch type: {type(batch)}, skipping")
                    continue
                
                if images is None or not isinstance(images, torch.Tensor):
                    logger.warning(f"Could not extract images tensor from batch {batch_idx}, skipping")
                    continue
                
                images = images.to(self.device)
                
                try:
                    # Forward pass through visual encoder
                    x = visual.conv1(images.type(visual.conv1.weight.dtype))
                    x = x.reshape(x.shape[0], x.shape[1], -1)
                    x = x.permute(0, 2, 1)
                    x = torch.cat([
                        visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                        x
                    ], dim=1)
                    x = x + visual.positional_embedding.to(x.dtype)
                    x = visual.ln_pre(x)
                    x = x.permute(1, 0, 2)
                    x = visual.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = visual.ln_post(x[:, 0, :])
                    
                    # Final projection
                    if visual.proj is not None:
                        final_proj = x @ visual.proj
                    else:
                        final_proj = x
                    final_features.append(final_proj.detach().cpu())
                    
                except Exception as e:
                    logger.warning(f"Forward pass failed for batch {batch_idx}: {e}")
                    continue
                
                if (batch_idx + 1) * self.config.batch_size >= self.config.num_images_per_split:
                    break
        
        # Remove hooks
        for hook in hook_registry.values():
            hook.remove()
        
        # Concatenate features
        result = {}
        for loc in hook_locations:
            if loc == 'final_proj':
                if final_features:
                    result[loc] = torch.cat(final_features, dim=0).numpy()[:self.config.num_images_per_split]
                else:
                    result[loc] = None
            elif features[loc]:
                result[loc] = torch.cat(features[loc], dim=0).numpy()[:self.config.num_images_per_split]
            else:
                result[loc] = None
        
        # Cache features
        if cache_path:
            np.save(cache_path, result)
            logger.info(f"Cached features to {cache_path}")
        
        return result
    
    def compute_cka(
        self,
        features_1: np.ndarray,
        features_2: np.ndarray,
        unbiased: bool = True
    ) -> float:
        """Compute Centered Kernel Alignment (CKA) between two feature sets."""
        # Center features
        f1 = features_1 - features_1.mean(axis=0, keepdims=True)
        f2 = features_2 - features_2.mean(axis=0, keepdims=True)
        
        # Compute Gram matrices
        gram_1 = f1 @ f1.T
        gram_2 = f2 @ f2.T
        
        # Compute CKA
        if unbiased:
            # Unbiased CKA (Kornblith et al., 2019)
            n = f1.shape[0]
            h = np.eye(n) - np.ones((n, n)) / n
            gram_1_centered = h @ gram_1 @ h
            gram_2_centered = h @ gram_2 @ h
            
            numerator = np.trace(gram_1_centered @ gram_2_centered)
            denominator = np.sqrt(np.trace(gram_1_centered @ gram_1_centered) * 
                                  np.trace(gram_2_centered @ gram_2_centered))
        else:
            # Linear CKA
            numerator = np.trace(gram_1 @ gram_2)
            denominator = np.sqrt(np.trace(gram_1 @ gram_1) * np.trace(gram_2 @ gram_2))
        
        cka = numerator / (denominator + 1e-10)
        return float(cka)
    
    def compute_cka_matrix(
        self,
        features_dict_1: Dict[str, np.ndarray],
        features_dict_2: Dict[str, np.ndarray],
        layer_names: List[str]
    ) -> np.ndarray:
        """Compute CKA matrix between all layers of two models."""
        n_layers = len(layer_names)
        cka_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                if layer_i in features_dict_1 and layer_j in features_dict_2:
                    if features_dict_1[layer_i] is not None and features_dict_2[layer_j] is not None:
                        # Ensure same number of samples
                        n_samples = min(features_dict_1[layer_i].shape[0], 
                                      features_dict_2[layer_j].shape[0])
                        f1 = features_dict_1[layer_i][:n_samples]
                        f2 = features_dict_2[layer_j][:n_samples]
                        
                        cka_matrix[i, j] = self.compute_cka(f1, f2)
        
        return cka_matrix
    
    def compute_svcca(
        self,
        features_1: np.ndarray,
        features_2: np.ndarray,
        n_components: Optional[int] = None,
        threshold: float = 0.99
    ) -> float:
        """Compute SVCCA (Singular Vector CCA) similarity between two feature sets.
        
        Args:
            features_1: Features from first model [n_samples, n_features1]
            features_2: Features from second model [n_samples, n_features2]
            n_components: Number of components to use (None = auto-determine from threshold)
            threshold: Variance threshold for component selection (0.99 = keep 99% variance)
        
        Returns:
            SVCCA similarity score (mean of canonical correlations)
        """
        # Center features
        f1 = features_1 - features_1.mean(axis=0, keepdims=True)
        f2 = features_2 - features_2.mean(axis=0, keepdims=True)
        
        # SVD on both feature sets
        U1, s1, V1t = np.linalg.svd(f1, full_matrices=False)
        U2, s2, V2t = np.linalg.svd(f2, full_matrices=False)
        
        # Determine number of components
        if n_components is None:
            # Use threshold to determine components
            cumvar1 = np.cumsum(s1**2) / np.sum(s1**2)
            cumvar2 = np.cumsum(s2**2) / np.sum(s2**2)
            n1 = np.searchsorted(cumvar1, threshold) + 1
            n2 = np.searchsorted(cumvar2, threshold) + 1
            n_components = min(n1, n2, min(f1.shape[1], f2.shape[1]))
        
        n_components = min(n_components, U1.shape[1], U2.shape[1])
        
        # Project to top n_components
        U1_reduced = U1[:, :n_components]
        U2_reduced = U2[:, :n_components]
        
        # Compute CCA between U1 and U2
        # CCA: find directions that maximize correlation
        C = U1_reduced.T @ U2_reduced  # Cross-covariance
        
        # SVD of cross-covariance gives canonical correlations
        _, d, _ = np.linalg.svd(C, full_matrices=False)
        
        # Canonical correlations are the singular values
        canonical_correlations = d
        
        # SVCCA similarity is mean of canonical correlations
        svcca_score = np.mean(canonical_correlations)
        
        return float(svcca_score)
    
    def compute_per_layer_cka(
        self,
        features_dict_1: Dict[str, np.ndarray],
        features_dict_2: Dict[str, np.ndarray],
        layer_names: List[str]
    ) -> pd.DataFrame:
        """Compute per-layer CKA between corresponding layers of two models."""
        results = []
        
        for layer_name in layer_names:
            if layer_name in features_dict_1 and layer_name in features_dict_2:
                f1 = features_dict_1[layer_name]
                f2 = features_dict_2[layer_name]
                
                if f1 is not None and f2 is not None:
                    # Ensure same number of samples
                    n_samples = min(f1.shape[0], f2.shape[0])
                    f1 = f1[:n_samples]
                    f2 = f2[:n_samples]
                    
                    cka = self.compute_cka(f1, f2, unbiased=True)
                    
                    results.append({
                        'layer': layer_name,
                        'cka': cka,
                        'n_samples': n_samples,
                        'dim1': f1.shape[1],
                        'dim2': f2.shape[1],
                    })
        
        return pd.DataFrame(results)
    
    def compute_per_layer_svcca(
        self,
        features_dict_1: Dict[str, np.ndarray],
        features_dict_2: Dict[str, np.ndarray],
        layer_names: List[str],
        n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute per-layer SVCCA between corresponding layers of two models."""
        results = []
        
        for layer_name in layer_names:
            if layer_name in features_dict_1 and layer_name in features_dict_2:
                f1 = features_dict_1[layer_name]
                f2 = features_dict_2[layer_name]
                
                if f1 is not None and f2 is not None:
                    # Ensure same number of samples
                    n_samples = min(f1.shape[0], f2.shape[0])
                    f1 = f1[:n_samples]
                    f2 = f2[:n_samples]
                    
                    svcca = self.compute_svcca(f1, f2, n_components=n_components)
                    
                    results.append({
                        'layer': layer_name,
                        'svcca': svcca,
                        'n_samples': n_samples,
                        'dim1': f1.shape[1],
                        'dim2': f2.shape[1],
                    })
        
        return pd.DataFrame(results)
    
    def train_linear_probe(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Train a linear probe and return metrics."""
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        test_features_scaled = scaler.transform(test_features)
        
        # Train logistic regression
        lr = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=1
        )
        lr.fit(features_scaled, labels)
        
        # Evaluate
        train_pred = lr.predict(features_scaled)
        test_pred = lr.predict(test_features_scaled)
        
        train_acc = accuracy_score(labels, train_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        
        # Calibration (ECE)
        test_probs = lr.predict_proba(test_features_scaled)
        ece = self._compute_ece(test_probs, test_labels)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'ece': ece,
        }
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        if probs.shape[1] == 2:
            probs = probs[:, 1]  # Binary classification
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (labels[in_bin] == (probs[in_bin] > 0.5)).mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)


class Visualizer:
    """Create visualizations for analysis results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
    
    def plot_parameter_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        title: str,
        filename: str,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """Plot heatmap of parameter changes by block and submodule."""
        # Pivot table: blocks × submodules
        pivot_data = df.pivot_table(
            values=metric,
            index='block_idx',
            columns='submodule',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': metric},
            ax=ax
        )
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Submodule', fontsize=12)
        ax.set_ylabel('Block Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved heatmap to {filename}")
    
    def plot_cka_heatmap(
        self,
        cka_matrix: np.ndarray,
        layer_names: List[str],
        title: str,
        filename: str,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot CKA heatmap between layers."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Shorten layer names for display
        short_names = [name.split('.')[-1] if '.' in name else name for name in layer_names]
        
        sns.heatmap(
            cka_matrix,
            xticklabels=short_names,
            yticklabels=short_names,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'CKA'},
            ax=ax
        )
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Model 2 Layers', fontsize=12)
        ax.set_ylabel('Model 1 Layers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved CKA heatmap to {filename}")
    
    def plot_interpolation_curves(
        self,
        alphas: np.ndarray,
        id_accs: Dict[str, np.ndarray],
        ood_accs: Dict[str, Dict[str, np.ndarray]],
        filename: str,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot weight interpolation curves for ID and OOD datasets."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ID accuracy
        ax = axes[0]
        for name, accs in id_accs.items():
            ax.plot(alphas, accs, marker='o', label=name, linewidth=2)
        ax.set_xlabel('Interpolation Coefficient α', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('ID Accuracy (ImageNet Val)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # OOD accuracy
        ax = axes[1]
        for dataset_name, methods in ood_accs.items():
            for method_name, accs in methods.items():
                ax.plot(alphas, accs, marker='o', label=f"{method_name} ({dataset_name})", 
                       linewidth=2, alpha=0.7)
        ax.set_xlabel('Interpolation Coefficient α', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('OOD Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved interpolation curves to {filename}")


class WeightInterpolator:
    """Perform weight-space interpolation (WiSE-style) between models."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def interpolate_weights(
        self,
        state_dict_1: OrderedDict,
        state_dict_2: OrderedDict,
        alpha: float
    ) -> OrderedDict:
        """Interpolate between two state dicts: α * sd_2 + (1-α) * sd_1."""
        interpolated = OrderedDict()
        
        common_keys = set(state_dict_1.keys()) & set(state_dict_2.keys())
        for key in common_keys:
            w1 = state_dict_1[key].float()
            w2 = state_dict_2[key].float()
            
            if w1.shape != w2.shape:
                logger.warning(f"Shape mismatch for {key}, using w1")
                interpolated[key] = w1.clone()
            else:
                interpolated[key] = (1 - alpha) * w1 + alpha * w2
        
        return interpolated
    
    def evaluate_interpolated_model(
        self,
        base_model: CLIPEncoder,
        interpolated_sd: OrderedDict,
        dataloader: torch.utils.data.DataLoader,
        classification_head: Optional[nn.Module] = None
    ) -> float:
        """Evaluate an interpolated model on a dataset."""
        # Create a copy of the model
        model = copy.deepcopy(base_model)
        model.load_state_dict(interpolated_sd)
        model = model.to(self.device)
        model.eval()
        
        if classification_head is None:
            # Use zeroshot head (matching compare_models_and_gradcam.py pattern)
            args = build_min_args(
                model="ViT-B/16",  # Use OpenAI CLIP format (slash, not hyphen)
                device=str(self.device),
                dataset="ImageNet",
                template="openai_imagenet_template",
                data_location=os.path.expanduser('~/data')
            )
            classification_head = get_zeroshot_classifier(args, model.model)
        
        classification_head = classification_head.to(self.device)
        classification_head.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating interpolated model"):
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = None
                
                images = images.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                try:
                    features = model(images)
                    logits = classification_head(features)
                    
                    if labels is not None:
                        pred = logits.argmax(dim=1)
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy


class DatasetLoader:
    """Load datasets for analysis (matching compare_models_and_gradcam.py pattern)."""
    
    def __init__(self, config: AnalysisConfig, preprocess_fn, data_location: str):
        self.config = config
        self.preprocess_fn = preprocess_fn
        self.data_location = data_location
        self.datasets = {}
    
    def load_dataset(self, dataset_name: str, dataset_path: Optional[str] = None) -> Optional[torch.utils.data.DataLoader]:
        """Load a dataset using repo's dataset classes (matching compare_models_and_gradcam.py)."""
        # Map dataset names to repo's dataset class names
        dataset_class_map = {
            'imagenet_val': 'ImageNet',
            'imagenet_v2': 'ImageNetV2',
            'imagenet_a': 'ImageNetA',
            'imagenet_r': 'ImageNetR',
            'imagenet_s': 'ImageNetSketch',
            'objectnet': 'ObjectNet',
        }
        
        repo_dataset_name = dataset_class_map.get(dataset_name, dataset_name)
        
        try:
            # Use repo's dataset loading pattern (matching compare_models_and_gradcam.py)
            dataset_cls = getattr(datasets, repo_dataset_name, None)
            if dataset_cls is None:
                logger.warning(f"Dataset class {repo_dataset_name} not found, trying ImageFolder fallback")
                import torchvision.datasets as tv_datasets
                if not dataset_path or not os.path.exists(dataset_path):
                    logger.warning(f"Dataset {dataset_name} not found at {dataset_path}, skipping")
                    return None
                dataset_obj = tv_datasets.ImageFolder(dataset_path, transform=self.preprocess_fn)
            else:
                # Use repo's dataset class
                dataset_obj = dataset_cls(
                    self.preprocess_fn,
                    location=self.data_location,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers
                )
                # Get test loader (val loader)
                dataloader = dataset_obj.test_loader if hasattr(dataset_obj, 'test_loader') else dataset_obj.train_loader
                
                # Limit samples if needed
                if hasattr(dataset_obj, 'test_dataset'):
                    original_dataset = dataset_obj.test_dataset
                elif hasattr(dataset_obj, 'train_dataset'):
                    original_dataset = dataset_obj.train_dataset
                else:
                    original_dataset = None
                
                if original_dataset and len(original_dataset) > self.config.num_images_per_split:
                    indices = np.random.choice(len(original_dataset), self.config.num_images_per_split, replace=False)
                    subset = torch.utils.data.Subset(original_dataset, indices)
                    dataloader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        pin_memory=True,
                        persistent_workers=True if self.config.num_workers > 0 else False,
                    )
                
                logger.info(f"Loaded {dataset_name} ({repo_dataset_name}): {len(original_dataset) if original_dataset else 'unknown'} samples")
                return dataloader
            
            # Fallback: ImageFolder with manual dataloader
            if len(dataset_obj) > self.config.num_images_per_split:
                indices = np.random.choice(len(dataset_obj), self.config.num_images_per_split, replace=False)
                dataset_obj = torch.utils.data.Subset(dataset_obj, indices)
            
            dataloader = torch.utils.data.DataLoader(
                dataset_obj,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
            
            logger.info(f"Loaded {dataset_name}: {len(dataset_obj)} samples")
            return dataloader
        
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def load_all_datasets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Load all available datasets."""
        loaded_datasets = {}
        
        for name, path in self.config.datasets.items():
            if path:
                loader = self.load_dataset(name, dataset_path=path)
                if loader is not None:
                    loaded_datasets[name] = loader
        
        return loaded_datasets


def generate_report(config: AnalysisConfig, output_path: Path, metrics: Dict[str, Any]):
    """Generate markdown report and convert to PDF."""
    report_md = f"""# Checkpoint Architecture Analysis Report

## Setup & Reproducibility

**Model Variant:** ViT-B/16  
**Pretrained Source:** OpenAI CLIP (from clip/ folder)  
**Direct-FT Checkpoint:** `{config.ckpt_direct}`  
**POMP-FT Checkpoint:** `{config.ckpt_pomp}`  
**Random Seed:** {config.seed}  
**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

### Dataset Configuration
- Images per split: {config.num_images_per_split}
- Batch size: {config.batch_size}
- Available datasets: {', '.join(metrics.get('datasets', []))}

## Parameter-Delta Results

### Methodology
We compute the following metrics for each parameter tensor:
- **L2 Norm Delta:** ||ΔW||₂
- **Relative Delta:** ||ΔW||₂ / (||W_pretrained||₂ + ε)
- **Cosine Similarity:** cos(W₁, W₂)
- **Frobenius Norm:** ||ΔW||_F
- **Spectral Drift:** Mean absolute difference of top-k singular values

### Key Findings

{metrics.get('param_summary', 'Analysis in progress...')}

## Representation-Delta Results

### CKA Analysis
{metrics.get('cka_summary', 'CKA analysis not performed (use --skip_representation to enable)')}

### Linear Probe Results
{metrics.get('probe_summary', 'Linear probe analysis not performed')}

## Weight Interpolation (WiSE-style)

{metrics.get('interpolation_summary', 'Interpolation analysis not performed (use --skip_interpolation to enable)')}

## Key Takeaways

{metrics.get('takeaways', 'Analysis summary pending...')}

## Appendix

### Implementation Notes
- All metrics computed with float32 precision
- Features cached to disk for reproducibility
- GPU-aware with CPU fallback

### Full Tables
See `tables/` directory for detailed CSV files.

### Figures
See `figures/` directory for all visualizations.
"""
    
    # Save markdown
    report_path = output_path / "report.md"
    with open(report_path, 'w') as f:
        f.write(report_md)
    logger.info(f"Saved report to {report_path}")
    
    # Try to convert to PDF
    try:
        pdf_path = output_path / "report.pdf"
        # Try pandoc first
        result = subprocess.run(
            ['pandoc', str(report_path), '-o', str(pdf_path), '--pdf-engine=xelatex'],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"Converted report to PDF: {pdf_path}")
        else:
            logger.warning(f"PDF conversion failed: {result.stderr.decode()}")
    except Exception as e:
        logger.warning(f"Could not convert to PDF: {e}")


def main():
    """Main entry point for checkpoint analysis."""
    parser = argparse.ArgumentParser(description="Checkpoint and Architecture Analysis Tool")
    
    # Model paths
    parser.add_argument("--ckpt_direct", type=str, required=True,
                       help="Path to Direct-FT (FLYP) checkpoint")
    parser.add_argument("--ckpt_pomp", type=str, required=True,
                       help="Path to POMP-FT checkpoint")
    parser.add_argument("--pretrained_from_repo", action="store_true", default=True,
                       help="Build pretrained model from repo (default: True)")
    
    # Dataset paths
    parser.add_argument("--imagenet_val", type=str, default=None,
                       help="Path to ImageNet validation set")
    parser.add_argument("--imagenet_v2", type=str, default=None,
                       help="Path to ImageNet-V2 dataset")
    parser.add_argument("--imagenet_a", type=str, default=None,
                       help="Path to ImageNet-A dataset")
    parser.add_argument("--imagenet_r", type=str, default=None,
                       help="Path to ImageNet-R dataset")
    parser.add_argument("--imagenet_s", type=str, default=None,
                       help="Path to ImageNet-Sketch dataset")
    parser.add_argument("--objectnet", type=str, default=None,
                       help="Path to ObjectNet dataset")
    
    # Analysis options
    parser.add_argument("--output_dir", type=str, default="analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--num_images_per_split", type=int, default=10000,
                       help="Number of images per dataset split for feature extraction")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--skip_representation", action="store_true",
                       help="Skip representation-delta analysis (faster)")
    parser.add_argument("--skip_interpolation", action="store_true",
                       help="Skip weight interpolation analysis")
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Use cached artifacts if available (default: True)")
    parser.add_argument("--force_recompute", action="store_true", default=False,
                       help="Force recomputation even if cached artifacts exist (default: False)")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create config
    config = AnalysisConfig(
        pretrained_from_repo=args.pretrained_from_repo,
        ckpt_direct=args.ckpt_direct,
        ckpt_pomp=args.ckpt_pomp,
        output_dir=args.output_dir,
        num_images_per_split=args.num_images_per_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        datasets={
            'imagenet_val': args.imagenet_val,
            'imagenet_v2': args.imagenet_v2,
            'imagenet_a': args.imagenet_a,
            'imagenet_r': args.imagenet_r,
            'imagenet_s': args.imagenet_s,
            'objectnet': args.objectnet,
        }
    )
    
    # Create output directories
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    (output_path / "tables").mkdir(exist_ok=True)
    (output_path / "artifacts").mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Checkpoint Architecture Analysis")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_path}")
    
    try:
        # Load models
        loader = ModelLoader(config)
        
        logger.info("\n" + "=" * 80)
        logger.info("Loading Models")
        logger.info("=" * 80)
        
        # Use 'ViT-B/16' format to ensure OpenAI CLIP is used (not open_clip)
        model_pretrained = loader.build_pretrained_model("ViT-B/16")
        model_direct = loader.load_checkpoint(config.ckpt_direct, "ViT-B/16")
        model_pomp = loader.load_checkpoint(config.ckpt_pomp, "ViT-B/16")
        
        # Extract state dicts
        sd_pretrained = loader.extract_state_dict(model_pretrained)
        sd_direct = loader.extract_state_dict(model_direct)
        sd_pomp = loader.extract_state_dict(model_pomp)
        
        # Build layer map
        logger.info("\n" + "=" * 80)
        logger.info("Building Layer Map")
        logger.info("=" * 80)
        layer_mapper = LayerMapper(model_pretrained)
        layer_map_path = output_path / "artifacts" / "layer_map.json"
        layer_mapper.save_layer_map(str(layer_map_path))
        
        # Determine cache usage
        use_cache = args.use_cache and not args.force_recompute
        
        # Parameter-delta analysis
        logger.info("\n" + "=" * 80)
        logger.info("Parameter-Delta Analysis")
        logger.info("=" * 80)
        
        param_analyzer = ParameterAnalyzer(config)
        
        # Compare Pretrained vs Direct-FT
        param_csv_pre_direct = output_path / "tables" / "param_metrics_pre_vs_direct.csv"
        if use_cache and param_csv_pre_direct.exists():
            logger.info(f"Loading cached parameter metrics from {param_csv_pre_direct}")
            df_pre_vs_direct = pd.read_csv(param_csv_pre_direct)
        else:
            logger.info("Comparing Pretrained vs Direct-FT...")
            df_pre_vs_direct = param_analyzer.compute_parameter_metrics(
                sd_pretrained, sd_direct, "Pretrained", "Direct-FT"
            )
            df_pre_vs_direct.to_csv(param_csv_pre_direct, index=False)
        
        # Compare Pretrained vs POMP-FT
        param_csv_pre_pomp = output_path / "tables" / "param_metrics_pre_vs_pomp.csv"
        if use_cache and param_csv_pre_pomp.exists():
            logger.info(f"Loading cached parameter metrics from {param_csv_pre_pomp}")
            df_pre_vs_pomp = pd.read_csv(param_csv_pre_pomp)
        else:
            logger.info("Comparing Pretrained vs POMP-FT...")
            df_pre_vs_pomp = param_analyzer.compute_parameter_metrics(
                sd_pretrained, sd_pomp, "Pretrained", "POMP-FT"
            )
            df_pre_vs_pomp.to_csv(param_csv_pre_pomp, index=False)
        
        # Compare Direct-FT vs POMP-FT
        param_csv_direct_pomp = output_path / "tables" / "param_metrics_direct_vs_pomp.csv"
        if use_cache and param_csv_direct_pomp.exists():
            logger.info(f"Loading cached parameter metrics from {param_csv_direct_pomp}")
            df_direct_vs_pomp = pd.read_csv(param_csv_direct_pomp)
        else:
            logger.info("Comparing Direct-FT vs POMP-FT...")
            df_direct_vs_pomp = param_analyzer.compute_parameter_metrics(
                sd_direct, sd_pomp, "Direct-FT", "POMP-FT"
            )
            df_direct_vs_pomp.to_csv(param_csv_direct_pomp, index=False)
        
        logger.info("\nParameter-delta analysis complete!")
        logger.info(f"Results saved to {output_path / 'tables'}")
        
        # Create visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Creating Visualizations")
        logger.info("=" * 80)
        
        visualizer = Visualizer(output_path)
        
        # Plot parameter heatmaps
        for df, pair_name in [(df_pre_vs_direct, "pre_vs_direct"), 
                              (df_pre_vs_pomp, "pre_vs_pomp"),
                              (df_direct_vs_pomp, "direct_vs_pomp")]:
            if len(df) > 0 and 'block_idx' in df.columns and 'submodule' in df.columns:
                # Filter valid block indices
                df_valid = df[df['block_idx'].notna()].copy()
                if len(df_valid) > 0:
                    visualizer.plot_parameter_heatmap(
                        df_valid,
                        'relative_delta',
                        f'Relative Parameter Delta: {pair_name.replace("_", " ").title()}',
                        f'param_heatmap_rel_delta_{pair_name}.png'
                    )
                    visualizer.plot_parameter_heatmap(
                        df_valid,
                        'cosine_similarity',
                        f'Cosine Similarity: {pair_name.replace("_", " ").title()}',
                        f'param_heatmap_cosine_{pair_name}.png'
                    )
        
        # Representation-delta analysis
        metrics = {
            'datasets': [],
            'param_summary': f"""
- Total parameters compared: {len(df_pre_vs_direct)}
- Average relative delta (Pre vs Direct-FT): {df_pre_vs_direct['relative_delta'].mean():.4f}
- Average relative delta (Pre vs POMP-FT): {df_pre_vs_pomp['relative_delta'].mean():.4f}
- Average cosine similarity (Pre vs Direct-FT): {df_pre_vs_direct['cosine_similarity'].mean():.4f}
- Average cosine similarity (Pre vs POMP-FT): {df_pre_vs_pomp['cosine_similarity'].mean():.4f}
            """.strip()
        }
        
        if not args.skip_representation and config.datasets.get('imagenet_val'):
            logger.info("\n" + "=" * 80)
            logger.info("Representation-Delta Analysis")
            logger.info("=" * 80)
            
            # Load datasets
            # Extract data location from first available dataset path
            data_location = None
            for dataset_path in config.datasets.values():
                if dataset_path and os.path.exists(dataset_path):
                    # Get parent directory (typically datasets/data/)
                    data_location = str(Path(dataset_path).parent.parent)
                    break
            
            if data_location is None:
                data_location = os.path.expanduser('~/data')
            
            dataset_loader = DatasetLoader(config, model_pretrained.val_preprocess, data_location)
            dataloaders = dataset_loader.load_all_datasets()
            metrics['datasets'] = list(dataloaders.keys())
            
            if 'imagenet_val' in dataloaders:
                logger.info("Extracting features for per-layer CKA/SVCCA analysis...")
                repr_analyzer = RepresentationAnalyzer(config)
                
                # Get hook locations
                hook_locations = repr_analyzer.get_vision_hook_locations(model_pretrained)
                logger.info(f"Extracting features from {len(hook_locations)} locations: {hook_locations}")
                
                # Extract features for all three models
                artifacts_dir = output_path / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                
                # Determine cache usage
                use_cache = args.use_cache and not args.force_recompute
                
                logger.info("Extracting features from pretrained model...")
                features_pretrained = repr_analyzer.extract_features(
                    model_pretrained,
                    dataloaders['imagenet_val'],
                    hook_locations=hook_locations,
                    model_name="pretrained",
                    cache_path=str(artifacts_dir / "features_pretrained.npy"),
                    use_cache=use_cache
                )
                
                logger.info("Extracting features from Direct-FT model...")
                features_direct = repr_analyzer.extract_features(
                    model_direct,
                    dataloaders['imagenet_val'],
                    hook_locations=hook_locations,
                    model_name="direct_ft",
                    cache_path=str(artifacts_dir / "features_direct.npy"),
                    use_cache=use_cache
                )
                
                logger.info("Extracting features from POMP-FT model...")
                features_pomp = repr_analyzer.extract_features(
                    model_pomp,
                    dataloaders['imagenet_val'],
                    hook_locations=hook_locations,
                    model_name="pomp_ft",
                    cache_path=str(artifacts_dir / "features_pomp.npy"),
                    use_cache=use_cache
                )
                
                # Compute per-layer CKA
                logger.info("Computing per-layer CKA...")
                cka_csv_pre_direct = output_path / "tables" / "per_layer_cka_pre_vs_direct.csv"
                cka_csv_pre_pomp = output_path / "tables" / "per_layer_cka_pre_vs_pomp.csv"
                cka_csv_direct_pomp = output_path / "tables" / "per_layer_cka_direct_vs_pomp.csv"
                
                if use_cache and all(f.exists() for f in [cka_csv_pre_direct, cka_csv_pre_pomp, cka_csv_direct_pomp]):
                    logger.info("Loading cached CKA results from CSV files...")
                    cka_pre_vs_direct = pd.read_csv(cka_csv_pre_direct)
                    cka_pre_vs_pomp = pd.read_csv(cka_csv_pre_pomp)
                    cka_direct_vs_pomp = pd.read_csv(cka_csv_direct_pomp)
                else:
                    cka_pre_vs_direct = repr_analyzer.compute_per_layer_cka(
                        features_pretrained, features_direct, hook_locations
                    )
                    cka_pre_vs_pomp = repr_analyzer.compute_per_layer_cka(
                        features_pretrained, features_pomp, hook_locations
                    )
                    cka_direct_vs_pomp = repr_analyzer.compute_per_layer_cka(
                        features_direct, features_pomp, hook_locations
                    )
                    # Save results
                    cka_pre_vs_direct.to_csv(cka_csv_pre_direct, index=False)
                    cka_pre_vs_pomp.to_csv(cka_csv_pre_pomp, index=False)
                    cka_direct_vs_pomp.to_csv(cka_csv_direct_pomp, index=False)
                
                # Compute per-layer SVCCA
                logger.info("Computing per-layer SVCCA...")
                svcca_csv_pre_direct = output_path / "tables" / "per_layer_svcca_pre_vs_direct.csv"
                svcca_csv_pre_pomp = output_path / "tables" / "per_layer_svcca_pre_vs_pomp.csv"
                svcca_csv_direct_pomp = output_path / "tables" / "per_layer_svcca_direct_vs_pomp.csv"
                
                if use_cache and all(f.exists() for f in [svcca_csv_pre_direct, svcca_csv_pre_pomp, svcca_csv_direct_pomp]):
                    logger.info("Loading cached SVCCA results from CSV files...")
                    svcca_pre_vs_direct = pd.read_csv(svcca_csv_pre_direct)
                    svcca_pre_vs_pomp = pd.read_csv(svcca_csv_pre_pomp)
                    svcca_direct_vs_pomp = pd.read_csv(svcca_csv_direct_pomp)
                else:
                    svcca_pre_vs_direct = repr_analyzer.compute_per_layer_svcca(
                        features_pretrained, features_direct, hook_locations
                    )
                    svcca_pre_vs_pomp = repr_analyzer.compute_per_layer_svcca(
                        features_pretrained, features_pomp, hook_locations
                    )
                    svcca_direct_vs_pomp = repr_analyzer.compute_per_layer_svcca(
                        features_direct, features_pomp, hook_locations
                    )
                    # Save results
                    svcca_pre_vs_direct.to_csv(svcca_csv_pre_direct, index=False)
                    svcca_pre_vs_pomp.to_csv(svcca_csv_pre_pomp, index=False)
                    svcca_direct_vs_pomp.to_csv(svcca_csv_direct_pomp, index=False)
                
                # Create visualizations
                logger.info("Creating per-layer CKA/SVCCA visualizations...")
                
                # Plot per-layer CKA line plots
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # CKA plot
                ax = axes[0]
                layer_indices = range(len(cka_pre_vs_direct))
                ax.plot(layer_indices, cka_pre_vs_direct['cka'], marker='o', label='Pre vs Direct-FT', linewidth=2)
                ax.plot(layer_indices, cka_pre_vs_pomp['cka'], marker='s', label='Pre vs POMP-FT', linewidth=2)
                ax.plot(layer_indices, cka_direct_vs_pomp['cka'], marker='^', label='Direct-FT vs POMP-FT', linewidth=2)
                ax.set_xlabel('Layer Index', fontsize=12)
                ax.set_ylabel('CKA', fontsize=12)
                ax.set_title('Per-Layer CKA', fontsize=14, fontweight='bold')
                ax.set_xticks(layer_indices)
                ax.set_xticklabels(cka_pre_vs_direct['layer'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # SVCCA plot
                ax = axes[1]
                ax.plot(layer_indices, svcca_pre_vs_direct['svcca'], marker='o', label='Pre vs Direct-FT', linewidth=2)
                ax.plot(layer_indices, svcca_pre_vs_pomp['svcca'], marker='s', label='Pre vs POMP-FT', linewidth=2)
                ax.plot(layer_indices, svcca_direct_vs_pomp['svcca'], marker='^', label='Direct-FT vs POMP-FT', linewidth=2)
                ax.set_xlabel('Layer Index', fontsize=12)
                ax.set_ylabel('SVCCA', fontsize=12)
                ax.set_title('Per-Layer SVCCA', fontsize=14, fontweight='bold')
                ax.set_xticks(layer_indices)
                ax.set_xticklabels(svcca_pre_vs_direct['layer'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig_path = output_path / "figures" / "per_layer_cka_svcca.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
                plt.close()
                logger.info(f"Saved per-layer CKA/SVCCA plot to {fig_path}")
                
                # Generate summary
                avg_cka_pre_direct = cka_pre_vs_direct['cka'].mean()
                avg_cka_pre_pomp = cka_pre_vs_pomp['cka'].mean()
                avg_cka_direct_pomp = cka_direct_vs_pomp['cka'].mean()
                
                avg_svcca_pre_direct = svcca_pre_vs_direct['svcca'].mean()
                avg_svcca_pre_pomp = svcca_pre_vs_pomp['svcca'].mean()
                avg_svcca_direct_pomp = svcca_direct_vs_pomp['svcca'].mean()
                
                metrics['cka_summary'] = f"""
**Per-Layer CKA Results:**
- Average CKA (Pre vs Direct-FT): {avg_cka_pre_direct:.4f}
- Average CKA (Pre vs POMP-FT): {avg_cka_pre_pomp:.4f}
- Average CKA (Direct-FT vs POMP-FT): {avg_cka_direct_pomp:.4f}

**Per-Layer SVCCA Results:**
- Average SVCCA (Pre vs Direct-FT): {avg_svcca_pre_direct:.4f}
- Average SVCCA (Pre vs POMP-FT): {avg_svcca_pre_pomp:.4f}
- Average SVCCA (Direct-FT vs POMP-FT): {avg_svcca_direct_pomp:.4f}

See `tables/per_layer_cka_*.csv` and `tables/per_layer_svcca_*.csv` for detailed layer-wise results.
                """.strip()
                
                metrics['probe_summary'] = "Linear probe analysis available - use per-layer features from artifacts/"
            else:
                metrics['cka_summary'] = "ImageNet validation dataset not available"
                metrics['probe_summary'] = "ImageNet validation dataset not available"
        else:
            metrics['cka_summary'] = "Skipped (use --skip_representation to enable)"
            metrics['probe_summary'] = "Skipped (use --skip_representation to enable)"
        
        # Weight interpolation
        if not args.skip_interpolation and config.datasets.get('imagenet_val'):
            logger.info("\n" + "=" * 80)
            logger.info("Weight Interpolation Analysis")
            logger.info("=" * 80)
            
            interpolator = WeightInterpolator(config)
            alphas = np.linspace(0.0, 1.0, 11)
            
            # For now, just log that interpolation would be performed
            logger.info("Weight interpolation requires full evaluation pipeline")
            logger.info("See notebook or use existing wise_ft.py for interpolation")
            metrics['interpolation_summary'] = """
Weight interpolation analysis available via existing `wise_ft.py` script.
Recommended: Use `src/wise_ft.py` with both checkpoints for comprehensive interpolation curves.
            """.strip()
        else:
            metrics['interpolation_summary'] = "Skipped (use --skip_interpolation to enable)"
        
        # Generate summary statistics
        metrics['takeaways'] = f"""
1. **Parameter Changes:** Direct-FT shows {'higher' if df_pre_vs_direct['relative_delta'].mean() > df_pre_vs_pomp['relative_delta'].mean() else 'lower'} relative parameter changes compared to POMP-FT
2. **Cosine Similarity:** {'Direct-FT' if df_pre_vs_direct['cosine_similarity'].mean() > df_pre_vs_pomp['cosine_similarity'].mean() else 'POMP-FT'} maintains higher similarity to pretrained weights
3. **Top Changed Layers:** See parameter heatmaps for layer-wise analysis
        """.strip()
        
        # Save run metadata
        run_meta = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': asdict(config),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip() if os.path.exists('.git') else 'unknown',
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        }
        
        with open(output_path / "artifacts" / "run_meta.json", 'w') as f:
            json.dump(run_meta, f, indent=2)
        
        # Generate report
        logger.info("\n" + "=" * 80)
        logger.info("Generating Report")
        logger.info("=" * 80)
        generate_report(config, output_path, metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis Complete!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"- Tables: {output_path / 'tables'}")
        logger.info(f"- Figures: {output_path / 'figures'}")
        logger.info(f"- Artifacts: {output_path / 'artifacts'}")
        logger.info(f"- Report: {output_path / 'report.md'}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

