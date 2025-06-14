#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.1
bs=512
ts=0.0
method=flyp_ldreg
fp16=1

for ldreg_coef in 0.1
do
for ldreg_k in 64
do

python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr ${lr} --wd ${wd} --batch-size $bs \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} \
--ldreg_coef $ldreg_coef --ldreg_k $ldreg_k --ldreg_type l1 \
--wb_project "clip_finetune" --method $method --use_fp16 ${fp16} --run 1

done
done 