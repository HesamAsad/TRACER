#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.1
bs=512
ts=0.0
method=carot
fp16=1

for sd in 0.9
do
  for alpha_fd in 100.0 200.0 400.0 600.0 800.0 1000.0 2000.0 3000.0
  do

    python src/main.py \
      --train-dataset ImageNet --epochs 10 --lr ${lr} --wd ${wd} --batch-size ${bs} \
      --model ViT-B/16 --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
      --template openai_imagenet_template --save ./checkpoints/ \
      --data-location ./datasets/data/ --ft_data ./datasets/csv/imagenet.csv \
      --csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} --cross_fnorm 0.05 \
      --distil_coef ${sd} --l_orth_wv 0 --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
      --wb_project clip_finetune --method ${method} --use_fp16 ${fp16} --run 2 --ema_up_freq 0 \
      --alpha_fd ${alpha_fd} --alpha_cross_kd 0.0 --alpha_icl 0.0 --alpha_crd 1.0 --workers 32

  done
done
