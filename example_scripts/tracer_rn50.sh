#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=9e-6
wd=0.02
bs=512
ts=0.0
method=tracer_rn50
fp16=1
seed=0

for sd in 0.01
do
  for alpha_fd in 200.0
  do

    python src/main.py \
      --train-dataset ImageNet --epochs 10 --lr ${lr} --wd ${wd} --batch-size ${bs} \
      --model RN50 --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
      --template openai_imagenet_template --save ./checkpoints/ --seed ${seed} \
      --data-location ./datasets/data/ --ft_data ./datasets/csv/imagenet.csv \
      --csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} --cross_fnorm 0.01 \
      --distil_coef ${sd} --l_orth_wv 0 --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
      --wb_project clip_finetune --method ${method} --use_fp16 ${fp16} --run 2 --ema_up_freq 0 \
      --alpha_fd ${alpha_fd} --alpha_cross_kd 1.0 --alpha_icl 1.0 --alpha_crd 1.0 --workers 32

  done
done
