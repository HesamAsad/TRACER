# Checkpoint Architecture Analysis Tool

Comprehensive checkpoint and architecture analysis tool for comparing Pretrained, Direct-FT (FLYP), and POMP-FT CLIP models.

## Overview

This tool performs:

1. **Parameter-Delta Analysis**: Compare weights between models (L2 norms, cosine similarity, spectral drift)
2. **Representation-Delta Analysis**: CKA/SVCCA similarity matrices and linear probe evaluation
3. **Weight Interpolation**: WiSE-style interpolation curves for robustness analysis
4. **Visualizations**: Heatmaps, bar charts, and interpolation curves
5. **Scientific Report**: Auto-generated Markdown/PDF report

## Quick Start

### Basic Usage

```bash
python -m tools.checkpoint_arch_analysis \
    --ckpt_direct "/path/to/direct_ft/checkpoint_10.pt" \
    --ckpt_pomp "/path/to/pomp_ft/checkpoint_10.pt" \
    --imagenet_val "/path/to/imagenet/val" \
    --output_dir analysis
```

### Full Example with All Datasets

```bash
python -m tools.checkpoint_arch_analysis \
    --ckpt_direct "/data/gpfs/projects/punim1316/CaRot/checkpoints/ImageNet/flyp/ViT-B/16_ep10_BS512_WD0.1_LR1e-05_D0.0_OC0.0_run100/checkpoint_10.pt" \
    --ckpt_pomp "/data/gpfs/projects/punim1316/CaRot/checkpoints/ImageNet/carot/ViT-B/16_ep10_BS512_WD0.1_LR1e-05_D0.95_OC0.0_run1/checkpoint_10.pt" \
    --imagenet_val "/data/gpfs/projects/punim1316/CaRot/datasets/data/ILSVRC2012/val" \
    --imagenet_v2 "/data/gpfs/projects/punim1316/CaRot/datasets/data/ImageNetV2-matched-frequency" \
    --imagenet_a "/data/gpfs/projects/punim1316/CaRot/datasets/data/imagenet-a" \
    --imagenet_r "/data/gpfs/projects/punim1316/CaRot/datasets/data/imagenet-r" \
    --imagenet_s "/data/gpfs/projects/punim1316/CaRot/datasets/data/sketch" \
    --objectnet "/path/to/objectnet" \
    --output_dir analysis \
    --num_images_per_split 10000 \
    --batch_size 128 \
    --num_workers 8 \
    --device cuda:0 \
    --seed 42
```

### Skipping Expensive Analyses

For faster iteration (parameter analysis only):

```bash
python -m tools.checkpoint_arch_analysis \
    --ckpt_direct "/path/to/direct_ft/checkpoint_10.pt" \
    --ckpt_pomp "/path/to/pomp_ft/checkpoint_10.pt" \
    --skip_representation \
    --skip_interpolation \
    --output_dir analysis
```

## Output Structure

```
analysis/
├── figures/
│   ├── param_heatmap_rel_delta_pre_vs_direct.png
│   ├── param_heatmap_cosine_pre_vs_direct.png
│   ├── param_heatmap_rel_delta_pre_vs_pomp.png
│   └── ...
├── tables/
│   ├── param_metrics_pre_vs_direct.csv
│   ├── param_metrics_pre_vs_pomp.csv
│   └── param_metrics_direct_vs_pomp.csv
├── artifacts/
│   ├── layer_map.json
│   ├── run_meta.json
│   └── (cached features if representation analysis enabled)
├── report.md
└── report.pdf (if pandoc available)
```

## Command-Line Arguments

### Required
- `--ckpt_direct`: Path to Direct-FT (FLYP) checkpoint
- `--ckpt_pomp`: Path to POMP-FT checkpoint

### Optional Dataset Paths
- `--imagenet_val`: ImageNet validation set path
- `--imagenet_v2`: ImageNet-V2 dataset path
- `--imagenet_a`: ImageNet-A dataset path
- `--imagenet_r`: ImageNet-R dataset path
- `--imagenet_s`: ImageNet-Sketch dataset path
- `--objectnet`: ObjectNet dataset path

### Analysis Options
- `--output_dir`: Output directory (default: `analysis`)
- `--num_images_per_split`: Number of images per dataset (default: 10000)
- `--batch_size`: Batch size for feature extraction (default: 128)
- `--num_workers`: Dataloader workers (default: 8)
- `--device`: Device to use (default: `cuda:0`)
- `--seed`: Random seed (default: 42)
- `--skip_representation`: Skip representation-delta analysis
- `--skip_interpolation`: Skip weight interpolation analysis

## Output Files

### Tables (CSV)
- `param_metrics_pre_vs_direct.csv`: Parameter metrics comparing Pretrained vs Direct-FT
- `param_metrics_pre_vs_pomp.csv`: Parameter metrics comparing Pretrained vs POMP-FT
- `param_metrics_direct_vs_pomp.csv`: Parameter metrics comparing Direct-FT vs POMP-FT

Each CSV contains:
- `key`: Parameter name
- `layer_type`: Vision/Text/Global
- `block_idx`: Transformer block index (if applicable)
- `submodule`: Submodule type (attn.qkv, mlp.fc1, etc.)
- `l2_norm_delta`: L2 norm of weight difference
- `relative_delta`: Relative change metric
- `cosine_similarity`: Cosine similarity between weights
- `frobenius_delta`: Frobenius norm of difference
- `spectral_drift`: Spectral drift metric

### Figures
- Parameter heatmaps showing changes by block and submodule
- CKA heatmaps (if representation analysis enabled)
- Interpolation curves (if interpolation enabled)

### Artifacts
- `layer_map.json`: Canonical layer structure mapping
- `run_meta.json`: Run metadata (git commit, versions, config)
- Cached feature files (if representation analysis enabled)

## Report

The tool generates a comprehensive Markdown report (`report.md`) and attempts to convert it to PDF if `pandoc` is available.

The report includes:
1. Setup & Reproducibility
2. Parameter-Delta Results
3. Representation-Delta Results (if enabled)
4. Weight Interpolation Results (if enabled)
5. Key Takeaways
6. Appendix

## Notes

- **Model Loading**: The tool automatically detects and handles DataParallel wrappers, EMA checkpoints, and various checkpoint formats
- **Caching**: Features are cached to disk to avoid recomputation
- **GPU-aware**: Automatically falls back to CPU if GPU unavailable
- **Robust**: Handles missing keys, shape mismatches, and missing datasets gracefully

## Integration with Existing Tools

For weight interpolation analysis, you can also use the existing `src/wise_ft.py` script:

```bash
python src/wise_ft.py \
    --load /path/to/pretrained.pt,/path/to/finetuned.pt \
    --eval_datasets ImageNet,ImageNetV2,ImageNetA,ImageNetR \
    --alpha 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

## Troubleshooting

### Checkpoint Loading Issues
- Ensure checkpoints are saved CLIPEncoder objects (not just state dicts)
- Check for DataParallel prefixes (`module.`) - these are handled automatically
- Verify checkpoint paths are absolute or correct relative paths

### Memory Issues
- Reduce `--num_images_per_split` if running out of memory
- Reduce `--batch_size` for feature extraction
- Use `--skip_representation` to skip memory-intensive analyses

### Missing Datasets
- The tool gracefully skips missing datasets
- Only ImageNet validation is required for basic parameter analysis
- OOD datasets are optional

## Example Output

```
================================================================================
Checkpoint Architecture Analysis
================================================================================
Output directory: analysis

================================================================================
Loading Models
================================================================================
Building pretrained model: ViT-B/16
Loading checkpoint: /path/to/direct_ft/checkpoint_10.pt
Loading checkpoint: /path/to/pomp_ft/checkpoint_10.pt

================================================================================
Parameter-Delta Analysis
================================================================================
Comparing Pretrained vs Direct-FT...
Common keys: 1234, Only in Pretrained: 0, Only in Direct-FT: 0
Computing parameter metrics: 100%|████████████| 1234/1234 [00:45<00:00, 27.3it/s]
...

Analysis Complete!
Results saved to: analysis
- Tables: analysis/tables
- Figures: analysis/figures
- Artifacts: analysis/artifacts
- Report: analysis/report.md
```

