#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.2
bs=256
ts=0.0
method=carot
fp16=1

for sd in 0.1
do
  for oc in 0.05
  do

    python src/main.py \
      --train-dataset IWildCamIDVal --epochs 20 --lr ${lr} --wd ${wd} --batch-size ${bs} \
      --model ViT-B/16 --eval-datasets IWildCamIDVal,IWildCamID,IWildCamOOD \
      --template iwildcam_template --save ./checkpoints/ \
      --data-location ./datasets/data/ --ft_data ./datasets/csv/iwildcam_v2.0/train.csv \
      --csv-img-key filepath --csv-caption-key title --exp_name iwildcam/${method} --cross_fnorm 0.05 \
      --distil_coef ${sd} --l_orth_wv ${oc} --max_grad_norm 0 --grad_norm_multiplier 0 --warmup_length 500 \
      --wb_project clip_finetune --method ${method} --use_fp16 ${fp16} --run 2 --ema_up_freq 0 --alpha_fd 5000.0 --alpha_cross_kd 1.0 --alpha_icl 1.0

  done
done
