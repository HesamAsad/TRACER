#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.1
bs=512
ts=0.0
method=tracer
fp16=1
ema_up_freq=0

for beta in 0.2 0.7 0.9 1.0 1.5
do
  for sd in 0.9
  do
    for alpha_fd in 2000.0
    do
      for alpha_cross_kd in 1.0
      do
        for alpha_icl in 1.0
        do
          for alpha_crd in 1.0
          do
            python src/main.py \
              --train-dataset ImageNet --epochs 10 --lr ${lr} --wd ${wd} --batch-size ${bs} \
              --model ViT-B/16 --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
              --template openai_imagenet_template --save ./checkpoints/ \
              --data-location ./datasets/data/ --ft_data ./datasets/csv/imagenet.csv \
              --csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} --cross_fnorm 0.05 \
              --distil_coef ${sd} --l_orth_wv 0 --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
              --wb_project clip_finetune --method ${method} --use_fp16 ${fp16} --run 1 --ema_up_freq ${ema_up_freq} \
              --alpha_fd ${alpha_fd} --alpha_cross_kd ${alpha_cross_kd} --alpha_icl ${alpha_icl} --alpha_crd ${alpha_crd} --workers 32 \
              --beta ${beta}
          done
        done
      done
    done
  done
done
