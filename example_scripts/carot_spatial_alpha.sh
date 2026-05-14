#!/bin/bash
# Scenario alpha alone -- geodesic text-text mixing (template <-> spatial dense caption).
# U-shape Beta(0.2, 0.2), per-batch lambda; no Scenario beta.
# Mirrors the flags that produced your best ViT-L/14 baseline (run 2).

cd /data/gpfs/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

# Two-GPU DataParallel. The code wraps the model with `torch.nn.DataParallel` over
# all visible CUDA devices, so just exposing both is sufficient.
export CUDA_VISIBLE_DEVICES=0,1

module load foss/2022a
module load Python/3.10.4
source .venv/bin/activate

SPATIAL_JSONL=/data/gpfs/projects/punim1316/KUEA/outputs/imagenet_train_dense_captions_caprl.cleaned.jsonl

# --batch-size 192 is the GLOBAL batch; DataParallel splits it 96/96 across the two
# GPUs. To keep the same effective gradient as the previous single-GPU run at bs=192,
# leave it as-is. To use the freed memory for throughput, bump to --batch-size 384
# (192 per GPU) and scale --lr to ~1.13e-5 (sqrt) or ~1.6e-5 (linear).
python src/main.py \
    --train-dataset ImageNet --epochs 10 --lr 8e-6 --wd 0.05 --batch-size 192 \
    --model ViT-L/14 \
    --eval-datasets ImageNet \
    --template openai_imagenet_template \
    --save ./checkpoints/ --data-location ./datasets/data/ \
    --ft_data ./datasets/csv/imagenet.csv --csv-img-key filepath --csv-caption-key title \
    --exp_name ImageNet/vitl14_spatial_alpha_captionv0_beta_0.2_0.2 \
    --cross_fnorm 0.04 --distil_coef 1.1 --l_orth_wv 0 \
    --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
    --wb_project clip_finetune --method tracer --use_fp16 1 --run 0 \
    --ema_up_freq 0 --alpha_fd 2000.0 --alpha_cross_kd 1.0 --alpha_icl 1.0 --alpha_crd 1.0 \
    --workers 32 \
    --use-spatial-captions 1 \
    --spatial-captions-jsonl "${SPATIAL_JSONL}" \
    --alpha-tt-mix 0.2 --beta-tt-mix 0.2 --tt-per-sample 0 \
    --beta-mix-coef 0.0 \
    --sanity-check
