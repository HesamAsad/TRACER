#!/usr/bin/env python3
"""
Test script to demonstrate the layer freezing functionality.
This script shows how the new arguments work without running full training.
"""

import sys
import os
sys.path.append('src')

import torch
from clip import clip
from src.args import parse_arguments
from src.models.utils import apply_layer_freezing
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_mock_args():
    """Create mock arguments for testing"""
    class MockArgs:
        def __init__(self):
            self.freeze_text_encoder = False
            self.trainable_layers = -1
    
    return MockArgs()

def test_layer_freezing():
    """Test different layer freezing configurations"""
    
    print("=" * 80)
    print("TESTING LAYER FREEZING FUNCTIONALITY")
    print("=" * 80)
    
    # Load a CLIP model for testing
    print("\nLoading CLIP model...")
    model, _, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    
    # Create a mock model wrapper (like CLIPEncoder) that properly inherits from nn.Module
    class MockCLIPEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.model = clip_model
        
        def parameters(self, recurse=True):
            return self.model.parameters(recurse)
    
    clip_encoder = MockCLIPEncoder(model)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Default (everything trainable)",
            "freeze_text_encoder": False,
            "trainable_layers": -1
        },
        {
            "name": "Freeze entire text encoder (embeddings + transformers + projections)",
            "freeze_text_encoder": True,
            "trainable_layers": -1
        },
        {
            "name": "Keep only last 2 transformer layers trainable (+ embeddings/projections)",
            "freeze_text_encoder": False,
            "trainable_layers": 2
        },
        {
            "name": "Freeze text encoder + keep last 1 vision layer trainable",
            "freeze_text_encoder": True,
            "trainable_layers": 1
        },
        {
            "name": "Freeze everything except logit_scale (extreme fine-tuning)",
            "freeze_text_encoder": False,
            "trainable_layers": 0
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*20} SCENARIO {i+1}: {scenario['name']} {'='*20}")
        
        # Reset model (reload to ensure fresh state)
        model, _, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        
        # Create fresh mock encoder
        class MockCLIPEncoder(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.model = clip_model
            
            def parameters(self, recurse=True):
                return self.model.parameters(recurse)
        
        clip_encoder = MockCLIPEncoder(model)
        
        # Create args
        args = create_mock_args()
        args.freeze_text_encoder = scenario["freeze_text_encoder"]
        args.trainable_layers = scenario["trainable_layers"]
        
        # Apply layer freezing
        apply_layer_freezing(clip_encoder, args, logger)
        
        print(f"\nConfiguration:")
        print(f"  freeze_text_encoder: {args.freeze_text_encoder}")
        print(f"  trainable_layers: {args.trainable_layers}")

def show_model_structure():
    """Show the structure of CLIP model for understanding"""
    print("\n" + "=" * 80)
    print("CLIP MODEL STRUCTURE")
    print("=" * 80)
    
    model, _, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    
    print("\nText Encoder (Transformer) Structure:")
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
        text_layers = model.transformer.resblocks
        print(f"  Total transformer layers: {len(text_layers)}")
        for i, layer in enumerate(text_layers):
            print(f"    Layer {i}: {type(layer).__name__}")
    
    print("\nVision Encoder Structure:")
    if hasattr(model, 'visual'):
        visual = model.visual
        if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            vision_layers = visual.transformer.resblocks
            print(f"  Vision Transformer layers: {len(vision_layers)}")
            for i, layer in enumerate(vision_layers):
                print(f"    Layer {i}: {type(layer).__name__}")
        elif hasattr(visual, 'layer1'):
            print(f"  ResNet structure detected:")
            resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4']
            for layer_name in resnet_layers:
                if hasattr(visual, layer_name):
                    layer = getattr(visual, layer_name)
                    print(f"    {layer_name}: {len(layer)} blocks")

def show_usage_examples():
    """Show usage examples for the command line"""
    print("\n" + "=" * 80)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        {
            "description": "Default: All components trainable",
            "command": "python train.py --method carot"
        },
        {
            "description": "Freeze entire text encoder (embeddings + transformers + projections)",
            "command": "python train.py --method carot --freeze_text_encoder"
        },
        {
            "description": "Keep only last 2 transformer layers trainable (+ embeddings/projections)",
            "command": "python train.py --method carot --trainable_layers 2"
        },
        {
            "description": "Freeze text encoder completely + keep only last 1 vision layer trainable",
            "command": "python train.py --method carot --freeze_text_encoder --trainable_layers 1"
        },
        {
            "description": "Freeze everything except logit_scale (extreme fine-tuning)",
            "command": "python train.py --method carot --trainable_layers 0"
        },
        {
            "description": "Fine-tune only vision embeddings and projections (freeze all transformers)",
            "command": "python train.py --method carot --freeze_text_encoder --trainable_layers 0"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

if __name__ == "__main__":
    try:
        show_model_structure()
        test_layer_freezing()
        show_usage_examples()
        
        print(f"\n{'='*80}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("You can now use --freeze_text_encoder and --trainable_layers arguments")
        print("in your training commands.")
        print("="*80)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 