# Spatial Caption Supervision via Geodesic Mixing (Scenarios α + β)

This extension injects dense spatial captions into Tracer fine-tuning without touching the
Beta/EMA teacher, KD module, Cross F-norm, orthogonality, gradient-norm scheduler,
mixed-precision, or eval pipeline. All new code is gated behind `--use-spatial-captions`;
leaving the flag unset reproduces current training byte-for-byte.

Reference: Oh et al., *Geodesic Multi-Modal Mixup for Robust Fine-Tuning* (NeurIPS 2023),
`github.com/changdaeoh/multimodal-mixup`. Our `sph_inter` convention and the
"overwrite-diagonal" pattern for Scenario β come from their `loss.py`.

Tracer = **T**rajectory-**R**obust **A**nchoring for **C**ontrastive **E**ncoder **R**egularization.

## Quick start (ViT-L/14, what to run next)

```bash
module load foss/2022a
module load Python/3.10.4
source .venv/bin/activate

# 1. Offline sanity (no GPU, few seconds):
python -m src.models.geodesic_mix

# 2. Scenario α alone (recommended first experimental run):
bash example_scripts/carot_spatial_alpha.sh
```

Then, if α beats the run-2 baseline on OOD averages, add Scenario β:

```bash
bash example_scripts/carot_spatial_alpha_beta.sh
```

Both scripts set `--sanity-check`, which asserts the α/β invariants on the first
batch (unit norm, endpoint recovery) and logs β hardness, then continues training.

The JSONL file `/data/gpfs/projects/punim1316/KUEA/outputs/imagenet_train_dense_captions_caprl_prompt2.jsonl`
has been verified against `./datasets/csv/imagenet.csv` — **100%** match rate on
1.28M entries (suffix fallback handles the `ILSVRC2012/` prefix divergence).

## What it does

- **Scenario α** — pair each image with a geodesic blend of its template caption and its
  dense spatial caption on the unit sphere, with λ ~ Beta(α, β). U-shaped Beta (α, β < 1)
  puts most mass near the endpoints so template-based zero-shot survives while spatial
  information still gets injected.
- **Scenario β** — build image-text geodesic hard negatives, plug them into the
  contrastive denominator off-diagonal, keep the original unmixed positives on the
  diagonal. Positive signal stays clean; negatives get harder by bridging the modality gap.

### Memory cost vs baseline

Per step, the α path runs the image encoder **once** and the text encoder **twice**
(on template tokens and spatial tokens). The second text encode bypasses DataParallel
and calls `core_model.encode_text` directly to avoid a redundant image-encoder pass.
Net overhead on ViT-L/14: a small amount of extra text-encoder activations
(~hundreds of MB at `batch-size=224`) — not a full extra forward. If you still see
OOM, drop `--batch-size` by 16–32 relative to your baseline and keep everything else.

## New flags (in `src/args.py`)

| Flag | Default | Purpose |
|---|---|---|
| `--use-spatial-captions` | `0` | Master switch. `0` = inert. |
| `--spatial-captions-jsonl` | `None` | Path to the JSONL file (`{"image": ..., "dense_caption": ...}`). |
| `--alpha-tt-mix` | `0.2` | α of `Beta(α, β)` for Scenario α text-text mixing. |
| `--beta-tt-mix` | `0.2` | β of `Beta(α, β)` for Scenario α. |
| `--tt-per-sample` | `0` | 1 = one λ per sample; 0 = one λ per batch (reference default). |
| `--beta-mix-coef` | `0.0` | Weight β_mix for Scenario β. 0 = α-only. |
| `--alpha-it-mix` | `0.5` | α of symmetric `Beta(α, α)` for Scenario β. |
| `--it-per-sample` | `0` | Per-sample vs per-batch λ for Scenario β. |
| `--beta-mix-target` | `spatial` | One of `{spatial, template, alpha_mixed}`. |
| `--tau2` | `0.0` | Literal `m_tau` multiplier for β **negatives** (off-diagonal). Positives on the diagonal always use the learned logit scale. Reference default `m_tau=0.01`. `<=0` reuses `lscale` for both. |
| `--sanity-check` | `False` | Assert α/β invariants on the first batch, then continue training. |

## Example commands

Environment:
```bash
module load foss/2022a
module load Python/3.10.4
source .venv/bin/activate
```

### Scenario α alone (U-shape Beta, per-batch λ)

```bash
python src/main.py --train-dataset ImageNet --epochs 10 --lr 8e-6 --wd 0.05 --batch-size 224 \
    --model ViT-L/14 \
    --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
    --template openai_imagenet_template \
    --save ./checkpoints/ --data-location ./datasets/data/ \
    --ft_data ./datasets/csv/imagenet.csv --csv-img-key filepath --csv-caption-key title \
    --exp_name ImageNet/carot_spatial_alpha \
    --cross_fnorm 0.04 --distil_coef 1.1 --l_orth_wv 0 \
    --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
    --wb_project clip_finetune --method carot --use_fp16 1 --run 3 \
    --ema_up_freq 0 --alpha_fd 2000.0 --alpha_cross_kd 1.0 --alpha_icl 1.0 --alpha_crd 1.0 \
    --workers 32 \
    --use-spatial-captions 1 \
    --spatial-captions-jsonl /data/gpfs/projects/punim1316/KUEA/outputs/imagenet_train_dense_captions_caprl_prompt2.jsonl \
    --alpha-tt-mix 0.2 --beta-tt-mix 0.2 --tt-per-sample 0 \
    --beta-mix-coef 0.0
```

### Scenario α + β (spatial mix target, separate β-stream temperature)

```bash
python src/main.py --train-dataset ImageNet --epochs 10 --lr 8e-6 --wd 0.05 --batch-size 224 \
    --model ViT-L/14 \
    --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
    --template openai_imagenet_template \
    --save ./checkpoints/ --data-location ./datasets/data/ \
    --ft_data ./datasets/csv/imagenet.csv --csv-img-key filepath --csv-caption-key title \
    --exp_name ImageNet/carot_spatial_alpha_beta \
    --cross_fnorm 0.04 --distil_coef 1.1 --l_orth_wv 0 \
    --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
    --wb_project clip_finetune --method carot --use_fp16 1 --run 4 \
    --ema_up_freq 0 --alpha_fd 2000.0 --alpha_cross_kd 1.0 --alpha_icl 1.0 --alpha_crd 1.0 \
    --workers 32 \
    --use-spatial-captions 1 \
    --spatial-captions-jsonl /data/gpfs/projects/punim1316/KUEA/outputs/imagenet_train_dense_captions_caprl_prompt2.jsonl \
    --alpha-tt-mix 0.2 --beta-tt-mix 0.2 --tt-per-sample 0 \
    --beta-mix-coef 0.5 --alpha-it-mix 0.5 --it-per-sample 0 \
    --beta-mix-target spatial --tau2 0.01
```

## Sanity checks

Add `--sanity-check` to either command. On the first batch the loop asserts:

1. **Unit norm** — `txt_feat_mixed.norm(dim=-1)` is within `[1 − 1e-2, 1 + 1e-2]`.
2. **Endpoint recovery** — `sph_inter(tmpl, sp, 1.0) == tmpl` and `sph_inter(tmpl, sp, 0.0) == sp` to 1e-2.
3. **Template fallback** — when `has_spatial=False` for a sample, its mixed vector equals its template vector (enforced by the `torch.where` mask).
4. **Scenario β hardness** — when `--beta-mix-coef > 0`, logs `mix_sim` vs `random_offdiag_sim` on the image-anchor diagonal. For Scenario β to help, `mix_sim` should be ≥ `random_offdiag_sim`.

The standalone unit tests for `sph_inter` live at the bottom of
`src/models/geodesic_mix.py`; run them with:

```bash
python -m src.models.geodesic_mix
```

## What is *not* modified

- `src/models/beta_moving_average.py`, EMA/BMA teacher logic
- `src/models/clip_knowledge_distillation.py`, `kd_module` forward
- Cross F-norm, orthogonality, gradient-norm scheduler
- OpenCLIP `ClipLoss`
- `get_zeroshot_classifier`, eval pipeline, checkpointing, wandb scaffolding
- `main.py` routing and existing `args.py` flags

## Metrics emitted

Per step (console + wandb, when `use_spatial=True`):
- `Spatial match rate`, `Alpha lambda mean` / `std`, `Text mixed-vs-template cos`, `Text mixed-vs-spatial cos`.
- When β is on: `Scenario-beta m2 loss`, `Scenario-beta weighted`.

Per epoch: `Avg Spatial Match Rate`, `Avg Scenario-beta m2 Loss`.

`CLIP Loss` in the logs is the **pure FLYP contrastive** term (Scenario-β contribution,
Cross F-norm, orthogonality, and distillation are all subtracted out — same convention
as before).

## File inventory

- `src/datasets_/spatial_captions.py` — JSONL loader, path normalization, match-rate log.
- `src/models/geodesic_mix.py` — `sph_inter` + `sample_beta_lambda` (+ self-test).
- `src/datasets_/laion.py` — `CsvDataset` / `get_csv_dataset` / `get_data` accept an optional `spatial_caption_index`; dataset yields a 4-tuple (or 5-tuple with labels) in the spatial path.
- `src/models/tracer_loss.py` — batch unpacking, Scenario α forward with mixed text features, Scenario β overwrite-diagonal contrastive loss, per-step and per-epoch logging, optional one-shot sanity asserts. All new code is behind `use_spatial`.
- `src/args.py` — 11 new flags (10 behavior + `--sanity-check`).
- `example_scripts/carot_spatial_alpha.sh`, `example_scripts/carot_spatial_alpha_beta.sh` — ready-to-run shell scripts mirroring your run-2 baseline flags.
