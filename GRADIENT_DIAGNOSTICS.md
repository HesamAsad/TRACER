# Gradient Diagnostics System

This document describes the deep gradient diagnostics system for CaRot training, designed to understand the natural dynamics of gradient components and their relationships during fine-tuning.

## Overview

The gradient diagnostics system tracks individual gradient components from different loss terms and analyzes their interactions, conflicts, and alignment with pre-trained weight directions. This provides insights into:

- **Gradient Component Dynamics**: How different loss terms (InfoNCE, Orthogonal Constraint, Cross F-norm, Self-Distillation) contribute to training
- **Gradient Conflicts**: When loss components work against each other
- **Pre-trained Alignment**: How gradients relate to moving towards/away from pre-trained weights
- **Training Stage Analysis**: How gradient patterns change during early, mid, and late training phases

## Key Features

### 1. Individual Gradient Tracking
Computes separate gradients for each loss component:
- **g_InfoNCE**: Gradients from the InfoNCE contrastive loss
- **g_OC**: Gradients from the orthogonality constraint loss
- **g_CrossF**: Gradients from the cross F-norm regularization
- **g_SD**: Gradients from the self-distillation loss

### 2. Reference Direction Analysis
Tracks direction vectors towards pre-trained weights:
- **d_pretrain = W_pretrain - W_current**: Direction to return to pre-trained state
- Analyzes how each gradient component aligns with this reference direction

### 3. Comprehensive Metrics
For each tracked parameter and training step:

#### Gradient Norms
- `||g_InfoNCE||`, `||g_OC||`, `||g_CrossF||`, `||g_SD||`
- `||d_pretrain||`: Distance from pre-trained weights
- `||g_total||`: Norm of combined gradient

#### Cosine Similarities
- **Component Alignments**: `cos(g_InfoNCE, g_OC)`, `cos(g_InfoNCE, g_CrossF)`, etc.
- **Pre-trained Alignments**: `cos(g_InfoNCE, d_pretrain)`, `cos(g_OC, d_pretrain)`, etc.
- **Total Gradient Alignment**: `cos(g_total, d_pretrain)`

### 4. Training Stage Breakdown
Automatically categorizes metrics by training progress:
- **Early stage**: First 33% of training
- **Mid stage**: 33-67% of training  
- **Late stage**: Final 33% of training

### 5. Conflict Detection
Identifies when gradient components work against each other:
- Detects cosine similarities < -0.3 as potential conflicts
- Tracks conflict patterns across training
- Provides summary statistics

## Tracked Components and Metrics

The system focuses on key CLIP model components and computes comprehensive gradient metrics for each tracked parameter $\theta_i$:

### Mathematical Formulation

For each tracked component $\theta_i$, we compute individual gradient components:
- $\mathbf{g}_{\text{InfoNCE}}^{(i)} = \nabla_{\theta_i} \mathcal{L}_{\text{InfoNCE}}$
- $\mathbf{g}_{\text{OC}}^{(i)} = \nabla_{\theta_i} \mathcal{L}_{\text{OC}}$ 
- $\mathbf{g}_{\text{CrossF}}^{(i)} = \nabla_{\theta_i} \mathcal{L}_{\text{CrossF}}$
- $\mathbf{g}_{\text{SD}}^{(i)} = \nabla_{\theta_i} \mathcal{L}_{\text{SD}}$

And reference direction: $\mathbf{d}_{\text{pretrain}}^{(i)} = \theta_i^{\text{pretrain}} - \theta_i^{\text{current}}$

### Tracked Components

#### Primary Targets (Projection Matrices)
- **Vision Projection ($W_v$)**: `model.visual.proj`
- **Text Projection ($W_l$)**: `model.text_projection`

#### Backbone Transformer Layers
- **Mid Vision Layer**: `model.visual.transformer.resblocks.6.*`
- **Last Vision Layer**: `model.visual.transformer.resblocks.11.*`
- **Mid Text Layer**: `model.transformer.resblocks.6.*`  
- **Last Text Layer**: `model.transformer.resblocks.11.*`

### Metric Patterns

For each tracked component $\theta_i$ with sanitized name `{component_name}`, we log:

#### Gradient Norms
$$\|\mathbf{g}_{\text{loss}}^{(i)}\|_2 \quad \text{for loss} \in \{\text{InfoNCE, OC, CrossF, SD}\}$$

**WandB Pattern**: `grad_diag/{component_name}_norm_{loss_name}`

**Examples**:
- `grad_diag/model_visual_proj_norm_infonce`
- `grad_diag/model_text_projection_norm_oc`
- `grad_diag/model_visual_transformer_resblocks_11_attn_in_proj_weight_norm_crossf`

#### Reference Direction Norms
$$\|\mathbf{d}_{\text{pretrain}}^{(i)}\|_2 = \|\theta_i^{\text{pretrain}} - \theta_i^{\text{current}}\|_2$$

**Physical Interpretation**: This measures the **Euclidean distance** the current parameters have moved from their pre-trained initialization. It quantifies **parameter drift magnitude**.

**Mathematical Details**:
- $\theta_i^{\text{pretrain}}$: Pre-trained parameter values (stored at initialization)
- $\theta_i^{\text{current}}$: Current parameter values during training
- $\mathbf{d}_{\text{pretrain}}^{(i)}$: Vector pointing **from current back to pre-trained** state

**Practical Significance**:
- **Small values** ($\|\mathbf{d}_{\text{pretrain}}\| \approx 0$): Conservative fine-tuning, staying close to pre-trained
- **Large values** ($\|\mathbf{d}_{\text{pretrain}}\| \gg 0$): Aggressive adaptation, significant drift from pre-trained
- **Rapid growth**: Potential catastrophic forgetting or unstable training

**Key Insights**:
1. **Drift Monitoring**: Track if model is moving too far from pre-trained capabilities
2. **Layer-wise Analysis**: Compare drift across different components (projection vs. attention layers)
3. **Training Phase Correlation**: Early training should show smaller drift, later training larger drift
4. **Performance Correlation**: Excessive drift may correlate with OOD performance degradation

**Critical Thresholds**:
- **Vision Projection ($W_v$)**: Drift > 1.0 may indicate significant visual representation change
- **Text Projection ($W_l$)**: Drift > 0.5 may indicate significant language representation change
- **Attention Layers**: Drift > 2.0 may indicate substantial architectural adaptation

**WandB Pattern**: `grad_diag/{component_name}_norm_d_pretrain`

**Examples**:
- `grad_diag/model_visual_proj_norm_d_pretrain`: How far $W_v$ has moved from pre-trained
- `grad_diag/model_text_projection_norm_d_pretrain`: How far $W_l$ has moved from pre-trained
- `grad_diag/model_visual_transformer_resblocks_11_attn_in_proj_weight_norm_d_pretrain`: Attention layer drift

**Analysis Questions**:
- Which components drift fastest? (Vision vs. Text vs. Attention)
- Does drift correlate with ID performance gains?
- Does excessive drift predict OOD performance drops?
- Are there optimal drift ranges for different components?

#### Pairwise Gradient Cosine Similarities
$$\cos(\mathbf{g}_{\text{loss1}}^{(i)}, \mathbf{g}_{\text{loss2}}^{(i)}) = \frac{\mathbf{g}_{\text{loss1}}^{(i)} \cdot \mathbf{g}_{\text{loss2}}^{(i)}}{\|\mathbf{g}_{\text{loss1}}^{(i)}\| \|\mathbf{g}_{\text{loss2}}^{(i)}\|}$$

**WandB Pattern**: `grad_diag/{component_name}_cos_sim_{loss1}_{loss2}`

**Examples**:
- `grad_diag/model_visual_proj_cos_sim_infonce_oc`
- `grad_diag/model_text_projection_cos_sim_infonce_crossf`
- `grad_diag/model_visual_proj_cos_sim_oc_sd`

#### Pre-trained Direction Cosine Similarities
$$\cos(\mathbf{g}_{\text{loss}}^{(i)}, \mathbf{d}_{\text{pretrain}}^{(i)}) = \frac{\mathbf{g}_{\text{loss}}^{(i)} \cdot \mathbf{d}_{\text{pretrain}}^{(i)}}{\|\mathbf{g}_{\text{loss}}^{(i)}\| \|\mathbf{d}_{\text{pretrain}}^{(i)}\|}$$

**WandB Pattern**: `grad_diag/{component_name}_cos_sim_{loss_name}_pretrain`

**Examples**:
- `grad_diag/model_visual_proj_cos_sim_infonce_pretrain`
- `grad_diag/model_text_projection_cos_sim_oc_pretrain`
- `grad_diag/model_visual_proj_cos_sim_total_pretrain`

**Relationship to Reference Direction Norms**:
The **combination** of reference direction norms and cosine similarities provides complete drift analysis:

$$\text{Parameter Change Vector: } \Delta\theta_i = \theta_i^{\text{current}} - \theta_i^{\text{pretrain}} = -\mathbf{d}_{\text{pretrain}}^{(i)}$$

**Joint Interpretation**:
- **Large** $\|\mathbf{d}_{\text{pretrain}}\|$ + **Negative** $\cos(\mathbf{g}, \mathbf{d}_{\text{pretrain}})$: 
  - *Rapid drift away from pre-trained, gradients accelerating the drift*
  - **Risk**: Catastrophic forgetting
  
- **Large** $\|\mathbf{d}_{\text{pretrain}}\|$ + **Positive** $\cos(\mathbf{g}, \mathbf{d}_{\text{pretrain}})$:
  - *Large drift but gradients trying to pull back to pre-trained*
  - **Interpretation**: Self-correction mechanism active
  
- **Small** $\|\mathbf{d}_{\text{pretrain}}\|$ + **Negative** $\cos(\mathbf{g}, \mathbf{d}_{\text{pretrain}})$:
  - *Conservative drift with gradients pushing for adaptation*
  - **Interpretation**: Controlled, gradual fine-tuning
  
- **Small** $\|\mathbf{d}_{\text{pretrain}}\|$ + **Positive** $\cos(\mathbf{g}, \mathbf{d}_{\text{pretrain}})$:
  - *Very conservative, gradients favor pre-trained state*
  - **Risk**: Insufficient adaptation, poor ID performance

#### Total Gradient Metrics
$$\mathbf{g}_{\text{total}}^{(i)} = \sum_{\text{loss}} \mathbf{g}_{\text{loss}}^{(i)}$$

**WandB Pattern**: `grad_diag/{component_name}_norm_total`

**Examples**:
- `grad_diag/model_visual_proj_norm_total`
- `grad_diag/model_text_projection_norm_total`

#### Training Stage Breakdowns
All metrics are additionally logged with stage prefixes:

**WandB Pattern**: `grad_diag/stage_{stage}_{metric_name}`

Where `{stage} ∈ \{\text{early}, \text{mid}, \text{late}\}$

**Examples**:
- `grad_diag/stage_early_model_visual_proj_norm_infonce`
- `grad_diag/stage_mid_model_visual_proj_cos_sim_infonce_oc`
- `grad_diag/stage_late_model_text_projection_cos_sim_total_pretrain`

### 📋 **Practical Example: Reference Direction Analysis**

Consider monitoring the **vision projection matrix** $W_v$ during CaRot training:

#### Step 1: Monitor Drift Magnitude
```
grad_diag/model_visual_proj_norm_d_pretrain = 0.85
```
**Interpretation**: $W_v$ has moved distance 0.85 from its pre-trained state (moderate drift)

#### Step 2: Check Gradient Alignment  
```
grad_diag/model_visual_proj_cos_sim_infonce_pretrain = -0.45
grad_diag/model_visual_proj_cos_sim_oc_pretrain = +0.62
```

**Interpretation**:
- **InfoNCE gradients** push **away** from pre-trained ($\cos = -0.45 < 0$)
- **Orthogonality gradients** pull **toward** pre-trained ($\cos = +0.62 > 0$)

#### Step 3: Combined Analysis
$$\text{Net Effect} = \lambda_{\text{InfoNCE}} \cdot (-0.45) + \lambda_{\text{OC}} \cdot (+0.62)$$

If $\lambda_{\text{InfoNCE}} = 1.0$ and $\lambda_{\text{OC}} = 0.2$:
$$\text{Net Effect} = 1.0 \cdot (-0.45) + 0.2 \cdot (+0.62) = -0.326$$

**Conclusion**: Net gradient pushes away from pre-trained, but orthogonality constraint provides **stabilizing force**

#### Step 4: Warning Signs
- If drift norm > 1.5: **High drift risk**
- If net cosine similarity < -0.7: **Catastrophic forgetting risk**  
- If InfoNCE dominates (ratio > 10:1): **Insufficient regularization**

## Usage

### 1. Enable Gradient Diagnostics

Add the following arguments to your training command:

```bash
python -m src.main \
    --method="carot" \
    --enable-grad-diagnostics \
    # ... other arguments
```

### 2. Arguments

- `--enable-grad-diagnostics`: Enable the gradient diagnostics system
- Diagnostics are logged at the same frequency as regular training logs (every 5 steps by default)

### 3. Output Metrics

All metrics are logged to WandB under the `grad_diag/` prefix:

```
grad_diag/visual_proj_norm_infonce
grad_diag/visual_proj_norm_oc  
grad_diag/visual_proj_cos_sim_infonce_oc
grad_diag/visual_proj_cos_sim_infonce_pretrain
grad_diag/stage_early_visual_proj_norm_infonce
grad_diag/performance_id_acc
grad_diag/performance_ood_acc
```

## Priority Screening Guide

Given the large number of metrics (827 plots), focus on these **high-priority patterns**:

### 🎯 **Tier 1: Core Research Questions (Focus Here First)**

#### Projection Matrix Dynamics ($W_v$, $W_l$)
```latex
\text{Focus on: } \begin{cases}
\text{grad\_diag/model\_visual\_proj\_norm\_*} \\
\text{grad\_diag/model\_text\_projection\_norm\_*}
\end{cases}
```

**Key Questions**: Which loss component $\mathcal{L}_i$ dominates? How do ratios $\frac{\|\mathbf{g}_{\text{InfoNCE}}\|}{\|\mathbf{g}_{\text{OC}}\|}$ evolve?

#### Gradient Conflict Detection
```latex
\text{Monitor: } \cos(\mathbf{g}_{\text{InfoNCE}}, \mathbf{g}_{\text{OC}}) < -0.3
```

**Specific Metrics**:
- `grad_diag/model_visual_proj_cos_sim_infonce_oc`
- `grad_diag/model_text_projection_cos_sim_infonce_oc`
- `grad_diag/model_visual_proj_cos_sim_infonce_crossf`
- `grad_diag/model_visual_proj_cos_sim_infonce_sd`

#### Pre-trained Alignment Analysis
```latex
\text{Track: } \cos(\mathbf{g}_{\text{loss}}, \mathbf{d}_{\text{pretrain}}) \begin{cases}
> 0 & \text{(conservative)} \\
< 0 & \text{(adaptive)} \\
\approx 0 & \text{(orthogonal)}
\end{cases}
```

**Specific Metrics**:
- `grad_diag/model_visual_proj_cos_sim_*_pretrain`
- `grad_diag/model_text_projection_cos_sim_*_pretrain`

### 🔍 **Tier 2: Training Dynamics (Check If Tier 1 Shows Issues)**

#### Stage-wise Evolution
```latex
\text{Compare: } \begin{cases}
\text{stage\_early\_*} & \text{(first 33\%)} \\
\text{stage\_mid\_*} & \text{(33-67\%)} \\
\text{stage\_late\_*} & \text{(final 33\%)}
\end{cases}
```

#### Total Gradient Behavior
```latex
\|\mathbf{g}_{\text{total}}\| = \left\|\sum_{i} \mathbf{g}_i\right\|
```

**Specific Metrics**:
- `grad_diag/model_visual_proj_norm_total`
- `grad_diag/model_visual_proj_cos_sim_total_pretrain`

### ⚠️ **Tier 3: Ignore Unless Debugging**

- Individual transformer components (`resblocks.*`)
- Layer normalization parameters (`ln_*`)
- Bias terms (`*_bias`)
- MLP internal weights (`mlp.c_fc`, `mlp.c_proj`)

### 📊 **Critical Alert Patterns**

#### Severe Conflicts
```latex
\cos(\mathbf{g}_i, \mathbf{g}_j) < -0.7 \quad \text{(Strong opposition)}
```

#### Rapid Pre-trained Drift
```latex
\frac{d}{dt}\cos(\mathbf{g}_{\text{total}}, \mathbf{d}_{\text{pretrain}}) < -0.1 \text{ per epoch}
```

#### Component Dominance Imbalance
```latex
\frac{\max_i \|\mathbf{g}_i\|}{\min_j \|\mathbf{g}_j\|} > 100 \quad \text{(One component overwhelms)}
```

## Example Analysis Workflow

### 1. Identify Dominant Components
```python
# Look for patterns in gradient norms
dominant_component = max(norm_metrics, key=lambda x: x['value'])
print(f"Dominant gradient: {dominant_component}")
```

### 2. Detect Conflicts
```python
# Find strong negative correlations
conflicts = [m for m in metrics if 'cos_sim_' in m and m['value'] < -0.5]
print(f"Gradient conflicts: {conflicts}")
```

### 3. Analyze Pre-trained Alignment
```python
# Check if gradients move towards or away from pre-trained weights
pretrain_alignments = [m for m in metrics if '_pretrain' in m]
moving_away = [m for m in pretrain_alignments if m['value'] < 0]
```

### 4. Stage-wise Analysis
```python
# Compare early vs late training dynamics
early_metrics = [m for m in metrics if 'stage_early_' in m]
late_metrics = [m for m in metrics if 'stage_late_' in m]
```

## Key Insights to Look For

### 1. Loss Component Dominance
- **Early Training**: InfoNCE typically dominates
- **Mid Training**: Regularization terms (OC, CrossF) may increase
- **Late Training**: Self-distillation often becomes more prominent

### 2. Gradient Conflicts
- **InfoNCE vs OC**: ID adaptation vs orthogonality preservation
- **InfoNCE vs SD**: Current batch vs teacher consistency
- **CrossF vs OC**: Cross-modal alignment vs within-modal orthogonality

### 3. Pre-trained Drift Analysis
- **Positive alignment**: Gradients point towards pre-trained weights (conservative)
- **Negative alignment**: Gradients point away from pre-trained weights (adaptation)
- **Near-zero alignment**: Orthogonal changes (specialized adaptation)

### 4. Performance Correlation
- **Conflict emergence**: Does gradient conflict timing correlate with OOD performance drops?
- **Alignment patterns**: Do certain alignment patterns predict better generalization?
- **Component balance**: What gradient norm ratios lead to best ID/OOD trade-offs?

## Troubleshooting

### Memory Issues
If you encounter memory issues:
- The system only tracks 6 key components by default
- Gradient computation uses `retain_graph=True` but cleans up afterwards
- Diagnostics use the same logging frequency as training (every 5 steps)

### Missing Metrics  
If some metrics don't appear:
- Check that loss components are non-zero (e.g., `--l-orth-wv > 0`)
- Ensure the model architecture matches expected patterns
- Verify WandB logging is enabled

### Performance Impact
The diagnostics system:
- Adds ~10-15% computational overhead when enabled
- Only computes diagnostics at specified intervals
- Uses efficient gradient computation with `autograd.grad`

## Advanced Usage

### Custom Component Tracking
To track additional components, modify `_get_tracked_components()` in `gradient_diagnostics.py`:

```python
key_patterns = [
    'visual.proj',  # Wv
    'text_projection',  # Wl
    'your.custom.layer',  # Add custom components
]
```

### Custom Analysis
Access the diagnostics history for custom analysis:

```python
# In your training script
if grad_diagnostics is not None:
    history = grad_diagnostics.metrics_history
    conflicts = grad_diagnostics.get_gradient_conflicts_summary()
    # Custom analysis code here
```

## Testing

Test the gradient diagnostics system:

```bash
python test_gradient_diagnostics.py --no-wandb
```

This will:
- Run a mock training loop with gradient diagnostics
- Generate visualization plots  
- Test all diagnostic metrics
- Verify conflict detection

## Files

- `src/models/gradient_diagnostics.py`: Main diagnostics implementation
- `test_gradient_diagnostics.py`: Test script and examples
- `example_scripts/carot_with_diagnostics.sh`: Usage examples
- `src/models/carot_loss.py`: Integration with CaRot training

## Citation

If you use the gradient diagnostics system in your research, please cite:

```bibtex
@article{carot_diagnostics,
  title={Deep Gradient Diagnostics for Multi-Objective CLIP Fine-tuning},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
``` 