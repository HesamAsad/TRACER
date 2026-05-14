import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')

def create_loss_landscape(W_pretrained, W_star, lambda_reg=0.5):
    """Create a convex loss landscape for visualization"""
    # Create grid
    x_range = np.linspace(-1.5, 2.5, 100)
    y_range = np.linspace(-1.5, 2.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute loss at each point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            w = np.array([X[i, j], Y[i, j]])
            # Task loss (quadratic)
            task_loss = 0.5 * np.linalg.norm(w - W_star)**2
            # Regularization loss (for static SD visualization)
            reg_loss = lambda_reg * 0.5 * np.linalg.norm(w - W_pretrained)**2
            Z[i, j] = task_loss
    
    return X, Y, Z

def beta_weight(t, T, beta=0.5):
    """Compute Beta(0.5, 0.5) weight for time t"""
    x = (t + 1) / (T + 2)  # Add small constants for stability
    # Beta(0.5, 0.5) PDF proportional to 1/sqrt(x(1-x))
    if x > 0 and x < 1:
        return 1.0 / (np.pi * np.sqrt(x * (1 - x)))
    return 0.0

def simulate_sd_bma(W_pretrained, W_star, T=50, lambda_reg=0.5):
    """Simulate SD-BMA trajectory"""
    # Initialize
    W_student = W_pretrained.copy()
    W_teacher = W_pretrained.copy()
    
    student_trajectory = [W_student.copy()]
    teacher_trajectory = [W_teacher.copy()]
    
    # Compute all alpha weights
    alphas = np.array([beta_weight(t, T) for t in range(T)])
    
    for t in range(T):
        # Student update (simplified for 2D visualization)
        a = lambda_reg / (1 + lambda_reg)
        W_student = a * W_teacher + (1 - a) * W_star
        
        # Teacher update with Beta moving average
        omega_t = alphas[t] / np.sum(alphas[:t+1])
        W_teacher = (1 - omega_t) * W_teacher + omega_t * W_student
        
        student_trajectory.append(W_student.copy())
        teacher_trajectory.append(W_teacher.copy())
    
    return np.array(student_trajectory), np.array(teacher_trajectory)

def simulate_static_sd(W_pretrained, W_star, T=50, lambda_reg=0.5):
    """Simulate static self-distillation trajectory"""
    W_current = W_pretrained.copy()
    trajectory = [W_current.copy()]
    
    a = lambda_reg / (1 + lambda_reg)
    
    for t in range(T):
        # Static SD converges to biased solution
        W_current = 0.9 * W_current + 0.1 * (a * W_pretrained + (1 - a) * W_star)
        trajectory.append(W_current.copy())
    
    return np.array(trajectory)

def create_visualization():
    """Create the main visualization"""
    # Set up parameters
    W_pretrained = np.array([0.5, 0.5])
    W_star = np.array([1.5, 1.5])
    lambda_reg = 0.8
    T = 30
    
    # Create loss landscape
    X, Y, Z = create_loss_landscape(W_pretrained, W_star, lambda_reg)
    
    # Simulate trajectories
    student_traj, teacher_traj = simulate_sd_bma(W_pretrained, W_star, T, lambda_reg)
    static_sd_traj = simulate_static_sd(W_pretrained, W_star, T, lambda_reg)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Main loss landscape plot
    ax1 = plt.subplot(1, 3, (1, 2))
    
    # Create contour plot
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = ax1.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.3, linewidths=0.5)
    contourf = ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Plot key points
    ax1.scatter(*W_pretrained, color='blue', s=200, marker='o', 
                label='Pretrained $W_I^0$', zorder=5, edgecolor='white', linewidth=2)
    ax1.scatter(*W_star, color='red', s=200, marker='*', 
                label='Task Solution $W^*_{FT}$', zorder=5, edgecolor='white', linewidth=2)
    
    # Plot static SD convergence point
    a = lambda_reg / (1 + lambda_reg)
    W_static_converge = a * W_pretrained + (1 - a) * W_star
    ax1.scatter(*W_static_converge, color='orange', s=150, marker='X', 
                label='Static SD Convergence', zorder=5, edgecolor='white', linewidth=2)
    
    # Plot trajectories
    ax1.plot(student_traj[:, 0], student_traj[:, 1], 'c-', linewidth=2.5, 
             label='SD-BMA Student', alpha=0.8)
    ax1.plot(teacher_traj[:, 0], teacher_traj[:, 1], 'm--', linewidth=2, 
             label='SD-BMA Teacher', alpha=0.8)
    ax1.plot(static_sd_traj[:, 0], static_sd_traj[:, 1], 'orange', linewidth=2, 
             label='Static SD', alpha=0.6, linestyle=':')
    
    # Add arrows to show direction
    for traj, color in [(student_traj, 'cyan'), (teacher_traj, 'magenta')]:
        for i in range(0, len(traj)-1, 5):
            if i > 0:
                ax1.annotate('', xy=traj[i+1], xytext=traj[i],
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
    
    # Styling
    ax1.set_xlabel('Weight Dimension 1', fontsize=12)
    ax1.set_ylabel('Weight Dimension 2', fontsize=12)
    ax1.set_title('SD-BMA: Adaptive Self-Distillation in Loss Landscape', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Loss Value', fontsize=10)
    
    # Beta distribution plot
    ax2 = plt.subplot(1, 3, 3)
    
    # Plot Beta(0.5, 0.5) distribution
    x_beta = np.linspace(0.001, 0.999, 1000)
    y_beta = 1 / (np.pi * np.sqrt(x_beta * (1 - x_beta)))
    ax2.fill_between(x_beta, y_beta, alpha=0.3, color='purple')
    ax2.plot(x_beta, y_beta, 'purple', linewidth=2)
    
    # Highlight early and late phases
    ax2.axvspan(0, 0.2, alpha=0.2, color='blue', label='Early Phase')
    ax2.axvspan(0.8, 1, alpha=0.2, color='red', label='Late Phase')
    
    ax2.set_xlabel('Training Progress (t/T)', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.set_title('Beta(0.5, 0.5) Weighting', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper center', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(y_beta) * 1.1)
    
    plt.suptitle('Self-Distillation with Beta Moving Average (SD-BMA)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def create_animated_visualization():
    """Create an animated version showing the evolution over time"""
    # Set up parameters
    W_pretrained = np.array([0.5, 0.5])
    W_star = np.array([1.5, 1.5])
    lambda_reg = 0.8
    T = 30
    
    # Create loss landscape
    X, Y, Z = create_loss_landscape(W_pretrained, W_star, lambda_reg)
    
    # Simulate trajectories
    student_traj, teacher_traj = simulate_sd_bma(W_pretrained, W_star, T, lambda_reg)
    static_sd_traj = simulate_static_sd(W_pretrained, W_star, T, lambda_reg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create static contour plot
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = ax.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.3, linewidths=0.5)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Plot static points
    ax.scatter(*W_pretrained, color='blue', s=200, marker='o', 
               label='Pretrained $W_I^0$', zorder=5, edgecolor='white', linewidth=2)
    ax.scatter(*W_star, color='red', s=200, marker='*', 
               label='Task Solution $W^*_{FT}$', zorder=5, edgecolor='white', linewidth=2)
    
    # Initialize lines
    student_line, = ax.plot([], [], 'c-', linewidth=2.5, label='SD-BMA Student', alpha=0.8)
    teacher_line, = ax.plot([], [], 'm--', linewidth=2, label='SD-BMA Teacher', alpha=0.8)
    static_line, = ax.plot([], [], 'orange', linewidth=2, label='Static SD', alpha=0.6, linestyle=':')
    
    # Initialize points
    student_point, = ax.plot([], [], 'co', markersize=10, zorder=6)
    teacher_point, = ax.plot([], [], 'mo', markersize=10, zorder=6)
    
    # Title and labels
    ax.set_xlabel('Weight Dimension 1', fontsize=12)
    ax.set_ylabel('Weight Dimension 2', fontsize=12)
    ax.set_title('SD-BMA: Adaptive Self-Distillation Animation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        student_line.set_data([], [])
        teacher_line.set_data([], [])
        static_line.set_data([], [])
        student_point.set_data([], [])
        teacher_point.set_data([], [])
        time_text.set_text('')
        return student_line, teacher_line, static_line, student_point, teacher_point, time_text
    
    def animate(frame):
        # Update trajectories
        student_line.set_data(student_traj[:frame+1, 0], student_traj[:frame+1, 1])
        teacher_line.set_data(teacher_traj[:frame+1, 0], teacher_traj[:frame+1, 1])
        static_line.set_data(static_sd_traj[:frame+1, 0], static_sd_traj[:frame+1, 1])
        
        # Update current points
        if frame < len(student_traj):
            student_point.set_data([student_traj[frame, 0]], [student_traj[frame, 1]])
            teacher_point.set_data([teacher_traj[frame, 0]], [teacher_traj[frame, 1]])
        
        # Update time text
        time_text.set_text(f'Step: {frame}/{T}')
        
        return student_line, teacher_line, static_line, student_point, teacher_point, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=T+1, 
                        interval=100, blit=True, repeat=True)
    
    return fig, anim

# Create and show the static visualization
fig = create_visualization()
fig.savefig('sd_bma_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Optionally create animated version (uncomment to use)
# fig_anim, anim = create_animated_visualization()
# plt.show()
# To save animation: anim.save('sd_bma_animation.gif', writer='pillow', fps=10)