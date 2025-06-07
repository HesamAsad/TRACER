#!/usr/bin/env python3
"""
Test script for Beta Moving Average implementation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.beta_moving_average import create_beta_weight_function

def test_beta_weight_function():
    """Test the Beta weight function visualization"""
    
    # Test parameters
    beta_values = [0.1, 0.5, 1.0, 2.0]
    total_iterations = 1000
    
    plt.figure(figsize=(12, 8))
    
    for i, beta in enumerate(beta_values):
        weight_func = create_beta_weight_function(beta, total_iterations)
        
        # Generate weights for all iterations
        iterations = np.arange(total_iterations)
        weights = [weight_func(it) for it in iterations]
        
        plt.subplot(2, 2, i+1)
        plt.plot(iterations, weights, label=f'Beta={beta}')
        plt.title(f'Beta Distribution Weights (β={beta})')
        plt.xlabel('Iteration')
        plt.ylabel('Weight')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('beta_weights_visualization.png', dpi=150, bbox_inches='tight')
    print("Beta weight visualization saved as 'beta_weights_visualization.png'")
    
    # Print some statistics
    print("\nBeta Weight Function Statistics:")
    for beta in beta_values:
        weight_func = create_beta_weight_function(beta, total_iterations)
        weights = [weight_func(it) for it in range(total_iterations)]
        print(f"Beta={beta}: Max={max(weights):.4f}, Min={min(weights):.4f}, "
              f"Mean={np.mean(weights):.4f}, Std={np.std(weights):.4f}")

def test_moving_average_behavior():
    """Test the moving average behavior with a simple model"""
    
    # Create a simple linear model
    model = torch.nn.Linear(10, 5)
    
    # Initialize with specific weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    
    # Create Beta moving average
    total_iterations = 100
    weight_func = create_beta_weight_function(0.5, total_iterations)
    
    from src.models.beta_moving_average import GeneralMovingAverage
    bma = GeneralMovingAverage(model, weight_func)
    
    print("\nTesting Moving Average Behavior:")
    print(f"Initial model weight: {model.weight[0, 0].item():.4f}")
    print(f"Initial BMA weight: {bma.moving_avg.weight[0, 0].item():.4f}")
    
    # Simulate training updates
    for i in range(10):
        # Modify model weights (simulate training)
        with torch.no_grad():
            model.weight += 0.1
            model.bias += 0.05
        
        # Update moving average
        bma.update()
        
        print(f"Step {i+1}: Model={model.weight[0, 0].item():.4f}, "
              f"BMA={bma.moving_avg.weight[0, 0].item():.4f}, "
              f"Weight={bma.weight:.6f}")

if __name__ == "__main__":
    print("Testing Beta Moving Average Implementation")
    print("=" * 50)
    
    try:
        test_beta_weight_function()
        test_moving_average_behavior()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 