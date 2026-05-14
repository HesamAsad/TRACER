#!/bin/bash
# Scenario alpha + beta -- adds image-text geodesic hard negatives on top of alpha.
# Positives on the diagonal keep `logit_scale`; negatives off-diagonal use --tau2 (= m_tau, reference default 0.01).

cd /data/gpfs/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

module load foss/2022a
module load Python/3.10.4
source .venv/bin/activate

SPATIAL_JSONL=/data/gpfs/projects/punim1316/KUEA/outputs/imagenet_train_dense_captions_caprl_prompt2.jsonl

python src/main.py \
    --train-dataset ImageNet --epochs 10 --lr 8e-6 --wd 0.05 --batch-size 224 \
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
    --spatial-captions-jsonl "${SPATIAL_JSONL}" \
    --alpha-tt-mix 0.2 --beta-tt-mix 0.2 --tt-per-sample 0 \
    --beta-mix-coef 0.5 --alpha-it-mix 0.5 --it-per-sample 0 \
    --beta-mix-target spatial --tau2 0.01 \
    --sanity-check
