#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.1
bs=512
ts=0.0
method=carot_ldreg
fp16=1
OC=0.0
SD=1.5
LDREG_COEF=0.2
LDREG_K=64
LDREG_TYPE="l1"

for sd in $SD
do
for oc in $OC
do


python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr $lr --wd $wd --batch-size $bs \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/$method \
--cross_fnorm 0.05 --l_orth_wv $oc --distil_coef $sd \
--ldreg_coef $LDREG_COEF --ldreg_k $LDREG_K --ldreg_type $LDREG_TYPE \
--wb_project "clip_finetune" --method $method --use_fp16 ${fp16} \
--run 2 
done
done
