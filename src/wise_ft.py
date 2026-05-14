import os
import wandb
import numpy as np

import torch
import pdb

from src.models.eval import evaluate
from src.models.ft_loss import finetune
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    """
    Robust merge of two state_dicts supporting missing keys and shape mismatches.
    - For keys present in both with same shape: weighted average (Fisher-weighted if provided).
    - For keys present in only one dict: copy from the present dict.
    - For mismatched shapes: copy from finetuned (theta_1) as default.
    """
    keys = set(theta_0.keys()) | set(theta_1.keys())
    out = {}

    fisher_0, fisher_1 = (None, None)
    if fishers is not None:
        fisher_0, fisher_1 = fishers

    for key in keys:
        v0 = theta_0.get(key, None)
        v1 = theta_1.get(key, None)

        if v0 is None and v1 is None:
            continue
        if v0 is None:
            out[key] = v1.clone()
            continue
        if v1 is None:
            out[key] = v0.clone()
            continue

        # Both present
        if v0.shape != v1.shape:
            # Prefer finetuned parameter when shapes differ
            out[key] = v1.clone()
            continue

        if fishers is None:
            out[key] = (1 - alpha) * v0 + alpha * v1
        else:
            ones = torch.ones_like(v0)
            f0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones) if fisher_0 is not None else ones
            f1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones) if fisher_1 is not None else ones
            c0 = (1 - alpha) * f0
            c1 = alpha * f1
            out[key] = (c0 * v0 + c1 * v1) / (c0 + c1)

    return out


def _ensure_defaults_for_zeroshot(args):
    # Provide defaults compatible with zeroshot head construction
    if getattr(args, 'train_dataset', None) is None:
        args.train_dataset = 'ImageNet'
    if getattr(args, 'template', None) is None:
        args.template = 'openai_imagenet_template'


def _load_image_classifier(path, args):
    """
    Load an ImageClassifier from a path that may contain either:
    - a pickled ImageClassifier (preferred)
    - a pickled CLIPEncoder, in which case build a zeroshot head aligned with current args
    """
    loaded = None
    # Try ImageClassifier first
    try:
        loaded = ImageClassifier.load(path)
    except Exception:
        loaded = None

    if isinstance(loaded, ImageClassifier):
        return loaded.to(args.device)

    # If the file actually contains a CLIPEncoder (common for TRACER checkpoints)
    if isinstance(loaded, CLIPEncoder):
        enc = loaded.to(args.device)
    else:
        try:
            enc = CLIPEncoder.load(path).to(args.device)
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint '{path}' as ImageClassifier or CLIPEncoder: {e}")

    # Build zeroshot head to wrap encoder into a classifier
    _ensure_defaults_for_zeroshot(args)
    head = get_zeroshot_classifier(args, enc.model)
    if hasattr(enc.model, 'transformer'):
        delattr(enc.model, 'transformer')
    clf = ImageClassifier(enc, head, process_images=False).to(args.device)
    return clf


def _construct_zeroshot_classifier_from_base(args):
    _ensure_defaults_for_zeroshot(args)
    enc = CLIPEncoder(args, keep_lang=True)
    head = get_zeroshot_classifier(args, enc.model)
    if hasattr(enc.model, 'transformer'):
        delattr(enc.model, 'transformer')
    clf = ImageClassifier(enc, head, process_images=False).to(args.device)
    return clf


def wise_ft(args):
    if args.wb_project:
        wandb_args = {'project': args.wb_project}
        wandb_args['name'] = args.method if args.method else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)
    assert args.save is not None, 'Please provide a path to store models'

    # Resolve inputs: prefer --load with two items (zeroshot, finetuned). Fallbacks supported.
    zeroshot_path, finetuned_path = None, None
    if getattr(args, 'load', None) is not None:
        if isinstance(args.load, list) and len(args.load) >= 2:
            zeroshot_path, finetuned_path = args.load[0], args.load[1]
        elif isinstance(args.load, str):
            # Single path provided; treat as finetuned, construct zeroshot from base
            finetuned_path = args.load
    if finetuned_path is None and getattr(args, 'clip_load', None):
        finetuned_path = args.clip_load

    # Load/construct zeroshot ImageClassifier
    if zeroshot_path is not None and os.path.exists(zeroshot_path):
        zeroshot = _load_image_classifier(zeroshot_path, args)
    else:
        zeroshot = _construct_zeroshot_classifier_from_base(args)

    # Load/construct finetuned ImageClassifier (from encoder or classifier)
    if finetuned_path is not None and os.path.exists(finetuned_path):
        finetuned = _load_image_classifier(finetuned_path, args)
    else:
        raise ValueError('Finetuned checkpoint path is required via --load or --clip_load.')
    

    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1


    alphas = args.alpha
    results_rows = []
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # Optionally save this merged model
        if args.save is not None:
            os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) != '' else None
            save_path = f"{args.save}_alpha{alpha:.2f}.pt"
            finetuned.save(save_path)

        epoch = 0
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        args.current_epoch = epoch
        
        finetuned.process_images = True
        eval_results = evaluate(finetuned, args, train_stats=epoch_stats)

        ood_acc, num_datasets, ood_ece = 0, 0, 0.0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy': continue
                ood_acc += v
            
            if 'ECE' in k:
                if k == 'ImageNet ECE': continue
                ood_ece += v
            num_datasets += 1/2
            
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
            ood_ece = ood_ece / num_datasets
        else:
            ood_acc, ood_ece = 0, 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        epoch_stats['Avg OOD ECE'] = round(ood_ece, 4)
        if wandb.run is not None:
            wandb.log({k:v for k, v in epoch_stats.items()})

        # Collect row for CSV
        row = {'alpha': alpha}
        for k, v in epoch_stats.items():
            try:
                if isinstance(v, (np.floating, np.integer)):
                    v = float(v)
                elif torch.is_tensor(v):
                    v = v.item()
            except Exception:
                pass
            row[k] = v
        results_rows.append(row)

    # Write CSV of all alphas
    try:
        import csv
        # Determine output path
        out_csv = None
        if getattr(args, 'results_db', None):
            base = args.results_db
            if base.endswith('.csv'):
                out_csv = base
            else:
                out_csv = os.path.splitext(base)[0] + '.csv'
        else:
            # default next to save path or cwd
            base_dir = os.path.dirname(args.save) if args.save else '.'
            out_csv = os.path.join(base_dir, 'wiseft_results.csv')

        # Build fieldnames from union of keys
        fieldnames = set()
        for r in results_rows:
            fieldnames.update(r.keys())
        fieldnames = ['alpha'] + sorted([f for f in fieldnames if f != 'alpha'])

        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) != '' else None
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_rows:
                writer.writerow(r)
        print(f"Saved wise-ft results to {out_csv}")
    except Exception as e:
        print(f"Warning: failed to write results CSV: {e}")

if __name__ == '__main__':
    args = parse_arguments()
    
    wise_ft(args)