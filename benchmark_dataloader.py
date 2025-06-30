#!/usr/bin/env python3
"""
Simple script to benchmark dataloader performance and identify bottlenecks.
"""

import time
import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from src.models.modeling import CLIPEncoder
from src.datasets_.laion import get_data
from src.args import parse_arguments
import src.datasets_ as datasets


def benchmark_dataloader(dataloader, name, num_batches=100):
    """Benchmark a single dataloader"""
    print(f"\n=== Benchmarking {name} ===")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Num workers: {dataloader.num_workers}")
    
    # Warmup
    print("Warming up...")
    iterator = iter(dataloader)
    for _ in range(min(5, num_batches)):
        try:
            batch = next(iterator)
            del batch  # Free memory immediately
        except StopIteration:
            break
    
    # Actual benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    iterator = iter(dataloader)
    
    for i in tqdm(range(num_batches), desc="Loading batches"):
        try:
            batch = next(iterator)
            # Simulate minimal processing
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # For image-text pairs
                if hasattr(batch[0], 'shape'):
                    _ = batch[0].shape  # Just access shape, don't move to GPU
                if hasattr(batch[1], 'shape'):
                    _ = batch[1].shape
            del batch  # Free memory immediately
        except StopIteration:
            print(f"DataLoader exhausted after {i} batches")
            break
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    batches_per_second = num_batches / total_time
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Batches per second: {batches_per_second:.2f}")
    print(f"Time per batch: {total_time/num_batches:.4f} seconds")
    
    return total_time, batches_per_second


def main():
    parser = argparse.ArgumentParser(description='Benchmark dataloader performance')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--num-batches', type=int, default=50, help='Number of batches to benchmark')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to test')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to test')
    parser.add_argument('--train-dataset', type=str, default='ImageNet', help='Dataset to benchmark')
    parser.add_argument('--data-location', type=str, required=True, help='Data location')
    parser.add_argument('--ft-data', type=str, help='Fine-tuning data path')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='Model for preprocessing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Add default values for required args
    parser.add_argument('--dataset-type', type=str, default='csv', help='Dataset type')
    parser.add_argument('--csv-img-key', type=str, default='filepath', help='Image key in CSV')
    parser.add_argument('--csv-caption-key', type=str, default='title', help='Caption key in CSV')
    parser.add_argument('--csv-separator', type=str, default='\t', help='CSV separator')
    
    benchmark_args = parser.parse_args()
    
    print("=== DataLoader Performance Benchmark ===")
    print(f"Testing with {benchmark_args.workers} workers, batch size {benchmark_args.batch_size}")
    
    # Initialize CLIP encoder for preprocessing
    clip_encoder = CLIPEncoder(benchmark_args, keep_lang=True)
    
    # Test main training dataset
    if benchmark_args.train_dataset:
        try:
            dataset_class = getattr(datasets, benchmark_args.train_dataset)
            dataset = dataset_class(
                clip_encoder.train_preprocess, 
                location=benchmark_args.data_location, 
                batch_size=benchmark_args.batch_size,
                num_workers=benchmark_args.workers
            )
            
            main_time, main_bps = benchmark_dataloader(
                dataset.train_loader, 
                f"Main Dataset ({benchmark_args.train_dataset})", 
                benchmark_args.num_batches
            )
        except Exception as e:
            print(f"Error benchmarking main dataset: {e}")
            main_time, main_bps = None, None
    
    # Test fine-tuning dataset if provided
    if benchmark_args.ft_data and os.path.exists(benchmark_args.ft_data):
        try:
            img_text_data = get_data(
                benchmark_args, 
                (clip_encoder.train_preprocess, clip_encoder.val_preprocess), 
                epoch=0
            )
            ft_dataloader = img_text_data["train_ft"].dataloader
            
            ft_time, ft_bps = benchmark_dataloader(
                ft_dataloader, 
                "Fine-tuning Dataset", 
                benchmark_args.num_batches
            )
        except Exception as e:
            print(f"Error benchmarking ft dataset: {e}")
            ft_time, ft_bps = None, None
    else:
        ft_time, ft_bps = None, None
    
    # Summary
    print("\n=== SUMMARY ===")
    if main_time is not None:
        print(f"Main dataset: {main_bps:.2f} batches/sec ({main_time:.2f}s total)")
    if ft_time is not None:
        print(f"FT dataset: {ft_bps:.2f} batches/sec ({ft_time:.2f}s total)")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    print("1. Ensure you're using all available CPU cores:")
    print(f"   - Current workers: {benchmark_args.workers}")
    print(f"   - Available CPU cores: {os.cpu_count()}")
    print("   - Recommended: Use --workers equal to number of CPU cores")
    
    print("\n2. Consider these optimizations:")
    print("   - Increase prefetch_factor (default 2)")
    print("   - Use persistent_workers=True")
    print("   - Set multiprocessing_context='spawn'")
    print("   - Enable pin_memory=True for GPU training")
    
    if main_bps is not None and main_bps < 1.0:
        print("\n⚠️  WARNING: Very slow dataloading detected!")
        print("   - Check if images are on fast storage (SSD vs HDD)")
        print("   - Consider caching preprocessed data")
        print("   - Verify image formats are efficiently readable")


if __name__ == "__main__":
    main() 