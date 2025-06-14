import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the scheduler
sys.path.append('src')
from models.utils import cosine_grad_norm_scheduler

def test_grad_norm_scheduler():
    # Test parameters (similar to your training setup)
    initial_grad_norm = 0.0001
    final_grad_norm = 0.001  # or initial_grad_norm * 10 for 10x increase
    
    # Simulate different training scenarios
    scenarios = [
        {"epochs": 10, "batch_size": 100, "name": "10 epochs, 100 batches"},
        {"epochs": 50, "batch_size": 100, "name": "50 epochs, 100 batches"},
        {"epochs": 100, "batch_size": 50, "name": "100 epochs, 50 batches"},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cosine Gradient Norm Scheduler Visualization', fontsize=16)
    
    # Plot each scenario
    for i, scenario in enumerate(scenarios):
        if i < 3:  # Only plot first 3 scenarios
            ax = axes[i//2, i%2]
            
            total_steps = scenario["epochs"] * scenario["batch_size"]
            scheduler = cosine_grad_norm_scheduler(initial_grad_norm, final_grad_norm, total_steps)
            
            steps = np.arange(total_steps)
            grad_norms = [scheduler(step) for step in steps]
            
            ax.plot(steps, grad_norms, 'b-', linewidth=2, label='Grad Norm')
            ax.axhline(y=initial_grad_norm, color='g', linestyle='--', alpha=0.7, label=f'Initial: {initial_grad_norm}')
            ax.axhline(y=final_grad_norm, color='r', linestyle='--', alpha=0.7, label=f'Final: {final_grad_norm}')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Max Gradient Norm')
            ax.set_title(scenario["name"])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add some statistics
            min_val = min(grad_norms)
            max_val = max(grad_norms)
            ax.text(0.02, 0.98, f'Min: {min_val:.6f}\nMax: {max_val:.6f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Fourth subplot: Compare different multipliers
    ax = axes[1, 1]
    total_steps = 1000
    multipliers = [2, 5, 10, 50, 100]
    
    for mult in multipliers:
        final_norm = initial_grad_norm * mult
        scheduler = cosine_grad_norm_scheduler(initial_grad_norm, final_norm, total_steps)
        
        steps = np.arange(total_steps)
        grad_norms = [scheduler(step) for step in steps]
        
        ax.plot(steps, grad_norms, linewidth=2, label=f'{mult}x increase')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Max Gradient Norm')
    ax.set_title('Different Multiplier Comparisons')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Log scale to see all curves clearly
    
    plt.tight_layout()
    plt.savefig('grad_norm_scheduler_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some specific values for verification
    print("\n" + "="*60)
    print("SCHEDULER VERIFICATION")
    print("="*60)
    
    scheduler = cosine_grad_norm_scheduler(initial_grad_norm, final_grad_norm, 1000)
    test_steps = [0, 100, 250, 500, 750, 999]
    
    print(f"Initial grad norm: {initial_grad_norm}")
    print(f"Final grad norm: {final_grad_norm}")
    print(f"Total steps: 1000")
    print("\nStep-by-step values:")
    
    for step in test_steps:
        value = scheduler(step)
        progress = step / 999 * 100
        print(f"Step {step:3d} ({progress:5.1f}%): {value:.6f}")
    
    print("\nFormula verification:")
    print("cosine_factor = 0.5 * (1 - cos(π * progress))")
    print("grad_norm = initial + (final - initial) * cosine_factor")
    
    for step in [0, 500, 999]:
        progress = step / 999
        cosine_factor = 0.5 * (1 - np.cos(np.pi * progress))
        expected = initial_grad_norm + (final_grad_norm - initial_grad_norm) * cosine_factor
        actual = scheduler(step)
        print(f"Step {step}: progress={progress:.3f}, cosine_factor={cosine_factor:.3f}, expected={expected:.6f}, actual={actual:.6f}")

if __name__ == "__main__":
    test_grad_norm_scheduler() 