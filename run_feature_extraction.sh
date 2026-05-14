#!/bin/bash

# Example script to run ImageNet feature extraction
# Modify the paths according to your setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "working directory: $PWD"
export PYTHONPATH="$PYTHONPATH:$PWD"

# Set paths
DATA_LOCATION="./datasets/data/"  # Root directory containing datasets
FT_DATA="./datasets/csv/imagenet_unique.csv"    # Path to ImageNet CSV file
OUTPUT_CSV="imagenet_features_vitb16.csv"

# Model configuration
MODEL="ViT-B/16"
BATCH_SIZE=512
WORKERS=4

echo "Starting ImageNet feature extraction..."
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_CSV"

python extract_imagenet_features.py \
    --data-location "$DATA_LOCATION" \
    --ft-data "$FT_DATA" \
    --model "$MODEL" \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --output-csv "$OUTPUT_CSV" \
    --template "openai_imagenet_template" \
    --train-dataset "ImageNet" \
    --csv-img-key "filepath" \
    --csv-caption-key "title" \
    --csv-separator "," \
    --device "cuda"

echo "Feature extraction completed!"
echo "Results saved to: $OUTPUT_CSV" 