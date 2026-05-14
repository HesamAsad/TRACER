# Model Comparison Report: DINOv2 vs CLIP vs TRACER

## Overall Performance Summary

### dinov2
- Average Silhouette Score: 0.068
- Average Davies-Bouldin Score: 3.544
- Average Within-class Similarity: 0.363
- Average Between-class Similarity: 0.043
- Separation (Within - Between): 0.320

### tracer
- Average Silhouette Score: 0.050
- Average Davies-Bouldin Score: 3.362
- Average Within-class Similarity: 0.678
- Average Between-class Similarity: 0.414
- Separation (Within - Between): 0.265

### clip_base
- Average Silhouette Score: 0.007
- Average Davies-Bouldin Score: 4.130
- Average Within-class Similarity: 0.704
- Average Between-class Similarity: 0.535
- Separation (Within - Between): 0.169

## Dataset-specific Performance

### imagenet_r
- Best Silhouette Score: dinov2 (0.063)
- Best Class Separation: dinov2 (0.246)

### imagenet_v2
- Best Silhouette Score: dinov2 (0.073)
- Best Class Separation: dinov2 (0.357)

### imagenet_sketch
- Best Silhouette Score: dinov2 (0.059)
- Best Class Separation: dinov2 (0.371)

### imagenet_val
- Best Silhouette Score: dinov2 (0.112)
- Best Class Separation: dinov2 (0.415)

### imagenet_a
- Best Silhouette Score: dinov2 (0.032)
- Best Class Separation: dinov2 (0.209)

## Key Findings

1. **Overall Best Model**: dinov2 with highest average silhouette score
2. **TRACER Performance**: Shows improvement over base CLIP (Silhouette: 0.050 vs 0.007)
3. **Generalization**: Performance varies significantly across different ImageNet variants
4. **Representation Quality**: DINOv2 consistently shows strong clustering properties

