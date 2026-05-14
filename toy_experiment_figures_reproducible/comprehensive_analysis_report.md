# Fine-tuning Method Comparison Report

**Generated on:** 2025-08-07 17:39:16

## Executive Summary

This report presents a comprehensive analysis comparing different fine-tuning strategies for multimodal contrastive learning models. The analysis includes:

1. **Embedding Space Analysis**: Visualization and statistical analysis of learned representations
2. **Decision Boundary Analysis**: Classification performance and separability metrics
3. **Feature Drift Analysis**: How representations change between original and spuriously correlated data

---

## 1. Embedding Statistics Summary

### 1.1 Within-class vs Between-class Similarity

| Method | Image Within-class | Image Between-class | Image Separation Ratio | Text Within-class | Text Between-class | Text Separation Ratio |
|--------|-------------------|-------------------|----------------------|------------------|------------------|---------------------|
| Pretrained Multimodal | 0.9097 | 0.0710 | 12.8155 | 0.9998 | 0.4223 | 2.3678 |
| Finetuned Direct | 0.9009 | 0.2454 | 3.6717 | 0.9998 | 0.4167 | 2.3994 |
| Finetuned L2Reg | 0.8962 | 0.2644 | 3.3900 | 0.9998 | 0.4236 | 2.3604 |
| Finetuned Selfdistill | 0.9076 | 0.2335 | 3.8865 | 0.9998 | 0.4185 | 2.3889 |
| Finetuned Dynamicdistill | 0.9084 | 0.2422 | 3.7504 | 0.9998 | 0.4176 | 2.3943 |

### 1.2 Embedding Norm Statistics

| Method | Image Embedding Norm | Text Embedding Norm |
|--------|---------------------|--------------------|
| Pretrained Multimodal | 10.5963 | 11.4167 |
| Finetuned Direct | 4.5790 | 11.4167 |
| Finetuned L2Reg | 4.0947 | 11.4167 |
| Finetuned Selfdistill | 4.7873 | 11.4167 |
| Finetuned Dynamicdistill | 4.7846 | 11.4167 |

---

## 2. Classification Performance

### 2.1 Image Embedding Classification

| Method | SVM | Logistic Regression | K-NN |
|--------|-----|-------------------|------|
| Pretrained Multimodal | 0.9744 | 0.9800 | 0.9767 |
| Finetuned Direct | 0.9822 | 0.9822 | 0.9811 |
| Finetuned L2Reg | 0.9822 | 0.9800 | 0.9844 |
| Finetuned Selfdistill | 0.9822 | 0.9811 | 0.9800 |
| Finetuned Dynamicdistill | 0.9833 | 0.9822 | 0.9833 |

### 2.2 Text Embedding Classification

| Method | SVM | Logistic Regression | K-NN |
|--------|-----|-------------------|------|
| Pretrained Multimodal | 1.0000 | 1.0000 | 1.0000 |
| Finetuned Direct | 1.0000 | 1.0000 | 1.0000 |
| Finetuned L2Reg | 1.0000 | 1.0000 | 1.0000 |
| Finetuned Selfdistill | 1.0000 | 1.0000 | 1.0000 |
| Finetuned Dynamicdistill | 1.0000 | 1.0000 | 1.0000 |

---

## 3. Embedding Separability Analysis

| Method | Image Separability Ratio | Text Separability Ratio |
|--------|--------------------------|------------------------|
| Pretrained Multimodal | 4.7243 | 64.9244 |
| Finetuned Direct | 3.4666 | 65.0366 |
| Finetuned L2Reg | 3.3130 | 64.9043 |
| Finetuned Selfdistill | 3.7981 | 64.9026 |
| Finetuned Dynamicdistill | 3.7198 | 65.0278 |

---

## 4. Feature Drift Analysis (Spurious Correlation Robustness)

### 4.1 Mean Cosine Similarity (Original vs Colored)

| Method | Image Similarity | Text Similarity |
|--------|-----------------|-----------------|
| Pretrained Multimodal | 1.0000 | 0.5859 |
| Finetuned Direct | 1.0000 | 0.5859 |
| Finetuned L2Reg | 1.0000 | 0.5859 |
| Finetuned Selfdistill | 1.0000 | 0.5859 |
| Finetuned Dynamicdistill | 1.0000 | 0.5859 |

### 4.2 Mean Class Centroid Drift

| Method | Image Drift | Text Drift |
|--------|------------|------------|
| Pretrained Multimodal | 0.0000 | 9.8423 |
| Finetuned Direct | 0.0000 | 9.8423 |
| Finetuned L2Reg | 0.0000 | 9.8423 |
| Finetuned Selfdistill | 0.0000 | 9.8423 |
| Finetuned Dynamicdistill | 0.0000 | 9.8423 |

---

## 5. Key Insights

### 5.1 Performance Summary

- **Best Image Classification**: Finetuned Dynamicdistill
- **Best Text Classification**: Pretrained Multimodal
- **Most Stable Image Embeddings**: Pretrained Multimodal
- **Most Stable Text Embeddings**: Pretrained Multimodal

### 5.2 Method-specific Observations

**Pretrained Multimodal**: Baseline pre-trained model without fine-tuning
- Shows excellent image embedding separability (ratio: 4.724)
- Highly robust to spurious correlations (similarity: 1.000)

**Finetuned Direct**: Direct fine-tuning with frozen text encoder
- Shows excellent image embedding separability (ratio: 3.467)
- Highly robust to spurious correlations (similarity: 1.000)

**Finetuned L2Reg**: L2 regularization to prevent drift from pre-trained weights
- Shows excellent image embedding separability (ratio: 3.313)
- Highly robust to spurious correlations (similarity: 1.000)

**Finetuned Selfdistill**: Self-distillation with static teacher
- Shows excellent image embedding separability (ratio: 3.798)
- Highly robust to spurious correlations (similarity: 1.000)

**Finetuned Dynamicdistill**: Dynamic self-distillation with EMA teacher
- Shows excellent image embedding separability (ratio: 3.720)
- Highly robust to spurious correlations (similarity: 1.000)


---

## 6. Generated Visualizations

The following visualization files were generated:

1. **tsne_embedding_spaces.png** - t-SNE visualization of embedding spaces
2. **pca_embedding_spaces.png** - PCA visualization of embedding spaces
3. **cross_modal_alignment.png** - Cross-modal similarity analysis
4. **embedding_statistics.png** - Statistical comparison of embeddings
5. **spurious_correlation_analysis.png** - Spurious correlation robustness
6. **decision_boundaries_2d.png** - 2D decision boundary visualization
7. **embedding_separability.png** - Separability metrics comparison
8. **classification_performance.png** - Classification performance comparison
9. **feature_drift.png** - Feature drift analysis

---

*Report generated by Fine-tuning Analysis Suite on 2025-08-07 17:39:16*
