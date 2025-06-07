#!/usr/bin/env python3
"""
Script to extract features, logits, and CLIP loss from ImageNet train set using pre-trained ViT-B/16 model.
Saves results to CSV file for analysis.
"""

import os
import csv
import time
import argparse
from tqdm.auto import tqdm
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from src.models.modeling import CLIPEncoder
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.laion import CsvDataset
from clip.loss import ClipLoss
import clip.clip as clip


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ImageNet features and logits')
    parser.add_argument('--data-location', type=str, required=True,
                        help='Root directory for datasets')
    parser.add_argument('--ft-data', type=str, required=True,
                        help='Path to ImageNet CSV file')
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='CLIP model architecture')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--output-csv', type=str, default='imagenet_features.csv',
                        help='Output CSV file path')
    parser.add_argument('--template', type=str, default='openai_imagenet_template',
                        help='Template for zero-shot classification')
    parser.add_argument('--train-dataset', type=str, default='ImageNet',
                        help='Dataset name for getting class names')
    parser.add_argument('--csv-img-key', type=str, default='filepath',
                        help='Image path key in CSV')
    parser.add_argument('--csv-caption-key', type=str, default='title',
                        help='Caption key in CSV')
    parser.add_argument('--csv-separator', type=str, default='\t',
                        help='CSV separator')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    return parser.parse_args()


class ImageNetFeatureExtractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize CLIP model
        print(f"Loading CLIP model: {args.model}")
        self.clip_encoder = CLIPEncoder(args, keep_lang=True)
        self.clip_model = self.clip_encoder.model.to(self.device)
        self.clip_model.eval()
        
        # Get zero-shot classifier
        print("Creating zero-shot classifier...")
        self.classification_head = get_zeroshot_classifier(args, self.clip_model)
        self.classification_head = self.classification_head.to(self.device)
        
        # Initialize CLIP loss function
        self.clip_loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
        )
        
        # Setup dataset
        print("Setting up dataset...")
        self.setup_dataset()
        
    def setup_dataset(self):
        """Setup the ImageNet dataset from CSV"""
        dataset = CsvDataset(
            input_filename=self.args.ft_data,
            transforms=self.clip_encoder.val_preprocess,  # Use validation preprocessing
            img_key=self.args.csv_img_key,
            caption_key=self.args.csv_caption_key,
            sep=self.args.csv_separator
        )
        
        # Limit samples if specified
        if self.args.max_samples is not None:
            dataset.images = dataset.images[:self.args.max_samples]
            dataset.captions = dataset.captions[:self.args.max_samples]
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(self.dataloader)}")
    
    def extract_features(self):
        """Extract features, logits, and CLIP loss for all images"""
        results = []
        
        print("Starting feature extraction...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing batches")):
                images, texts = batch
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                # Get image and text features from CLIP
                image_features, text_features, logit_scale = self.clip_model(images, texts)
                
                # Compute CLIP loss
                clip_loss, logits_per_image, logits_per_text = self.clip_loss_fn(
                    image_features, text_features, logit_scale
                )
                
                # Get zero-shot classification logits
                zeroshot_logits = self.classification_head(image_features)
                
                # Convert to numpy for easier handling
                image_features_np = image_features.cpu().numpy()
                text_features_np = text_features.cpu().numpy()
                zeroshot_logits_np = zeroshot_logits.cpu().numpy()
                logits_per_image_np = logits_per_image.cpu().numpy()
                
                # Compute per-sample CLIP loss (approximate)
                # Since CLIP loss is computed over the batch, we approximate per-sample loss
                per_sample_clip_loss = self.compute_per_sample_clip_loss(
                    image_features, text_features, logit_scale
                )
                
                # Store results for each sample in the batch
                batch_start_idx = batch_idx * self.args.batch_size
                for i in range(len(images)):
                    sample_idx = batch_start_idx + i
                    
                    # Get top-5 predictions
                    top5_indices = np.argsort(zeroshot_logits_np[i])[-5:][::-1]
                    # top5_probs = F.softmax(torch.from_numpy(zeroshot_logits_np[i]), dim=0)[top5_indices].numpy()
                    
                    result = {
                        'sample_idx': sample_idx,
                        'clip_loss': per_sample_clip_loss[i].item(),
                        'logit_scale': logit_scale.exp().item(),
                        'top1_class': top5_indices[0],
                        # 'top1_prob': top5_probs[0],
                        'top5_classes': ','.join(map(str, top5_indices)),
                        # 'top5_probs': ','.join(map(str, top5_probs)),
                        'max_logit': np.max(zeroshot_logits_np[i]),
                        'min_logit': np.min(zeroshot_logits_np[i]),
                        # 'logit_std': np.std(zeroshot_logits_np[i]),
                        'logits': ','.join(map(lambda x: f"{x:.3f}", zeroshot_logits_np[i])),
                        # 'image_feature_norm': np.linalg.norm(image_features_np[i]),
                        # 'text_feature_norm': np.linalg.norm(text_features_np[i]),
                        'image_text_similarity': np.dot(image_features_np[i], text_features_np[i])
                    }
                    
                    # Add individual logits for all classes (optional - can be memory intensive)
                    # for class_idx in range(len(zeroshot_logits_np[i])):
                    #     result[f'logit_class_{class_idx}'] = zeroshot_logits_np[i][class_idx]
                    
                    results.append(result)
        
        return results
    
    def compute_per_sample_clip_loss(self, image_features, text_features, logit_scale):
        """Compute approximate per-sample CLIP loss"""
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits_per_image = logit_scale * image_features @ text_features.T
        
        # Create labels (diagonal elements are positive pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Compute cross-entropy loss for each sample
        per_sample_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        
        return per_sample_loss
    
    def save_results(self, results):
        """Save results to CSV file"""
        print(f"Saving results to {self.args.output_csv}")
        
        df = pd.DataFrame(results)
        df.to_csv(self.args.output_csv, index=False)
        
        print(f"Saved {len(results)} samples to {self.args.output_csv}")
        
        # Print some statistics
        print("\nStatistics:")
        print(f"Average CLIP loss: {df['clip_loss'].mean():.4f}")
        print(f"Average top-1 probability: {df['top1_prob'].mean():.4f}")
        print(f"Average logit scale: {df['logit_scale'].mean():.4f}")
        print(f"Average image-text similarity: {df['image_text_similarity'].mean():.4f}")


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
    
    # Initialize feature extractor
    extractor = ImageNetFeatureExtractor(args)
    
    # Extract features
    start_time = time.time()
    results = extractor.extract_features()
    end_time = time.time()
    
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    extractor.save_results(results)
    
    print("Done!")


if __name__ == "__main__":
    main() 