# Misclassification Analysis with CLIP and VLM

This script analyzes misclassified samples from pre-trained CLIP models using a Vision-Language Model (VLM) to determine if the predicted class is actually present in the image.

## Overview

The script performs the following analysis:

1. **Loads a pre-trained CLIP model** (default: ViT-B/16)
2. **Processes multiple datasets** (ImageNet, ImageNetV2, ImageNetR, ImageNetA, ImageNetSketch, ObjectNet)
3. **Identifies misclassifications** where CLIP's prediction differs from the ground truth
4. **Uses a VLM** to check if the predicted class is actually present in the misclassified image
5. **Logs confidence scores** from both CLIP and VLM
6. **Generates detailed reports** with statistics and analysis

## Key Features

- **Multi-dataset support**: Analyzes all major ImageNet variants
- **Confidence tracking**: Records confidence scores from both CLIP and VLM
- **Detailed logging**: Saves comprehensive results to CSV files
- **Progress tracking**: Shows real-time progress with tqdm
- **Flexible configuration**: Customizable models, datasets, and parameters

## Installation

```bash
pip install -r requirements_analysis.txt
```

## Usage

### Basic Usage

```bash
python analyze_misclassifications.py \
    --data-location /path/to/your/datasets \
    --output-dir ./results \
    --max-batches 50
```

### Advanced Usage

```bash
python analyze_misclassifications.py \
    --clip-model ViT-L/14 \
    --vlm-model zai-org/GLM-4-9B-0414 \
    --datasets ImageNet ImageNetV2 ImageNetR \
    --data-location /path/to/your/datasets \
    --batch-size 16 \
    --max-batches 100 \
    --output-dir ./detailed_analysis \
    --device cuda
```

## Arguments

### Model Arguments
- `--clip-model`: CLIP model to use (default: "ViT-B/16")
- `--vlm-model`: VLM model to use (default: "zai-org/GLM-4-9B-0414")
- `--device`: Device to use (default: "cuda")

### Dataset Arguments
- `--datasets`: List of datasets to analyze (default: all ImageNet variants)
- `--data-location`: Path to dataset location
- `--batch-size`: Batch size for processing (default: 32)
- `--max-batches`: Maximum number of batches to process per dataset (default: 100)

### Output Arguments
- `--output-dir`: Output directory for results (default: "./misclassification_analysis")

## Output Files

The script generates two main output files:

1. **Detailed Results** (`misclassifications_detailed_TIMESTAMP.csv`):
   - Individual misclassification records
   - Image paths, true classes, predicted classes
   - CLIP and VLM confidence scores
   - VLM responses and reasoning

2. **Summary Results** (`misclassifications_summary_TIMESTAMP.csv`):
   - Per-dataset statistics
   - Agreement rates between CLIP and VLM
   - Average confidence scores

## Example Output

```
Dataset: ImageNet
  Total misclassifications: 1,250
  VLM agrees with CLIP: 850 (68.0%)
  VLM disagrees with CLIP: 400 (32.0%)
  Avg CLIP confidence: 0.723
  Avg VLM confidence: 0.815

Dataset: ImageNetV2
  Total misclassifications: 980
  VLM agrees with CLIP: 620 (63.3%)
  VLM disagrees with CLIP: 360 (36.7%)
  Avg CLIP confidence: 0.689
  Avg VLM confidence: 0.792
```

## Analysis Insights

This analysis helps understand:

1. **False Positive Analysis**: When CLIP predicts a class that's not present
2. **Confidence Calibration**: How well CLIP's confidence aligns with VLM assessment
3. **Dataset-specific Patterns**: Which datasets have more problematic misclassifications
4. **Model Agreement**: How often CLIP and VLM agree on misclassifications

## Notes

- The current implementation uses a simulated VLM response for demonstration
- For production use, replace the `VLMClassifier.ask_about_image()` method with a real multimodal model
- Consider using models like GPT-4V, LLaVA, or similar for actual image analysis
- The script is designed to be memory-efficient and can process large datasets in batches

## Troubleshooting

1. **Memory Issues**: Reduce batch size or max-batches
2. **CUDA Out of Memory**: Use smaller models or reduce batch size
3. **Dataset Path Issues**: Ensure data-location points to correct dataset directories
4. **Model Loading Issues**: Check internet connection for model downloads

## Future Improvements

- [ ] Integrate with real multimodal VLMs (GPT-4V, LLaVA, etc.)
- [ ] Add visualization of misclassified samples
- [ ] Implement confidence calibration analysis
- [ ] Add support for custom datasets
- [ ] Include more detailed error analysis 