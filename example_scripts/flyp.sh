#!/bin/bash

cd /data/projects/punim1316/CaRot
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

lr=1e-5
wd=0.1
bs=512
method=flyp
fp16=1

python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr ${lr} --wd ${wd} --batch-size $bs \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} \
--wb_project "clip_finetune" --method $method --use_fp16 ${fp16} --run 1