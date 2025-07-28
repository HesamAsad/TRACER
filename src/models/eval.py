import os
import json
from tqdm import tqdm
import torch
import numpy as np
from src.models import utils
from src.models.modeling import ImageClassifier
from src.models.utils import LabelSmoothing
from src.datasets_.common import get_dataloader, maybe_dictionarize
import src.datasets_ as datasets
import torch.nn.functional as F

from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score
import pdb

import src.visualize as visualize
import matplotlib.pyplot as plt
from src.models.utils import clip_img_preprocessing, attack_pgd
from clip.loss import ClipLoss

# Add sklearn imports for detailed classification metrics
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import seaborn as sns


def eval_single_dataset(
    image_classifier,
    dataset,
    args,
    classification_head,
    visualization=False,
    dataset_name=None,
):
    model = image_classifier
    input_key = "images"
    image_enc = None
    model.eval()

    if classification_head is None: 
        pass
    else: 
        classification_head.eval()

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    batched_data = enumerate(dataloader)
    device = args.device

    # pdb.set_trace()

    if hasattr(dataset, "post_loop_metrics"):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    
    top1, correct, n = 0., 0., 0.
    tot_ece = 0.

    ys, y_hats, confidences = torch.Tensor([]).to(device), torch.Tensor([]).to(device),torch.Tensor([]).to(device)

    for i, data in tqdm(batched_data, desc=f"Evaluating on {dataset_name}", total=len(dataloader)):

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        if 'image_paths' in data:
            image_paths = data['image_paths']
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.use_fp16 else torch.float32), torch.no_grad():
            logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        ece_metric = MulticlassCalibrationError(num_classes=logits.shape[1], n_bins=10, norm='l1')

        if args.temperature_scale > 0:
            logits = logits * args.temperature_scale

        # for reliabiltiy diagram
        prob = F.softmax(logits, dim=1)
        confidence, y_hat = torch.max(prob, axis=1)

        confidences = torch.cat((confidences, confidence))
        y_hats = torch.cat((y_hats, y_hat))
        ys = torch.cat((ys, y))

        if args.full_eval:
            pass
        else:
            tot_ece += ece_metric(prob, y) * logits.shape[0]

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data[
                'metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

    top1 = correct / n
    
    # Calculate final ECE
    if args.full_eval and ys.numel() > 0:
        # For full eval, compute ECE on all collected data
        ece_metric_final = MulticlassCalibrationError(num_classes=int(torch.max(ys).item())+1, n_bins=10, norm='l1')
        mean_ece = ece_metric_final(confidences, ys.long())
    else:
        mean_ece = tot_ece / n if n > 0 else 0.0

    # Compute macro F1 score and detailed classification metrics
    if ys.numel() > 0 and y_hats.numel() > 0:
        # Convert to numpy for sklearn metrics
        y_true = ys.cpu().numpy().astype(int)
        y_pred = y_hats.cpu().numpy().astype(int)
        
        # Compute macro F1
        f1_metric = MulticlassF1Score(num_classes=int(torch.max(ys).item())+1, average="macro").to(device)
        macro_f1 = f1_metric(y_hats.long(), ys.long()).item()
        
        # Compute confusion matrix
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        # Compute classification report
        class_report = classification_report(y_true, y_pred, labels=range(num_classes), 
                                           output_dict=True, zero_division=0)
        
        # Calculate class-wise accuracies for later use
        class_correct = np.diag(conf_matrix)
        class_totals = conf_matrix.sum(axis=1)
        class_accuracies = np.divide(class_correct, class_totals, out=np.zeros_like(class_correct, dtype=float), where=class_totals!=0)
        
        # Print detailed metrics (for all datasets - it's text-based)
        print(f"\n=== Classification Report for {dataset_name} ===")
        print(classification_report(y_true, y_pred, labels=range(num_classes), zero_division=0))
        
        # Print class-wise accuracy to identify potential class collapse
        
        if num_classes <= 50:  # For small datasets, show all classes
            print(f"\n=== Class-wise Accuracy for {dataset_name} ===")
            for class_idx in range(num_classes):
                if class_totals[class_idx] > 0:
                    class_acc = class_correct[class_idx] / class_totals[class_idx]
                    print(f"Class {class_idx}: {class_acc:.4f} ({class_correct[class_idx]}/{class_totals[class_idx]})")
                else:
                    print(f"Class {class_idx}: No samples")
        else:  # For large datasets, show summary statistics and extremes
            print(f"\n=== Class-wise Accuracy Summary for {dataset_name} ({num_classes} classes) ===")
            print(f"Mean class accuracy: {class_accuracies.mean():.4f} ± {class_accuracies.std():.4f}")
            print(f"Min class accuracy: {class_accuracies.min():.4f}")
            print(f"Max class accuracy: {class_accuracies.max():.4f}")
            print(f"Classes with 0% accuracy: {np.sum(class_accuracies == 0)}")
            print(f"Classes with 100% accuracy: {np.sum(class_accuracies == 1.0)}")
            print(f"Classes with <10% accuracy: {np.sum(class_accuracies < 0.1)}")
            print(f"Classes with >90% accuracy: {np.sum(class_accuracies > 0.9)}")
            
            # Show a few examples of worst and best classes (but don't duplicate if we're also doing wandb logging)
            if wandb.run is None:  # Only show if not logging to wandb (to avoid duplication)
                worst_classes = np.argsort(class_accuracies)[:5]  # 5 worst classes
                best_classes = np.argsort(class_accuracies)[-5:]  # 5 best classes
                
                print(f"\nWorst performing classes:")
                for i, class_idx in enumerate(worst_classes):
                    if class_totals[class_idx] > 0:
                        print(f"  Class {class_idx}: {class_accuracies[class_idx]:.4f} ({class_correct[class_idx]}/{class_totals[class_idx]})")
                
                print(f"\nBest performing classes:")
                for i, class_idx in enumerate(reversed(best_classes)):
                    if class_totals[class_idx] > 0:
                        print(f"  Class {class_idx}: {class_accuracies[class_idx]:.4f} ({class_correct[class_idx]}/{class_totals[class_idx]})")
        
        # Log to wandb if available
        try:
            if wandb.run is not None:
                # === VISUAL COMPONENTS (size-dependent) ===
                if num_classes <= 50:  # Only visualize confusion matrix for small datasets
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {dataset_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    # Log confusion matrix image to wandb
                    wandb.log({f"{dataset_name}/confusion_matrix": wandb.Image(plt)})
                    plt.close()
                else:
                    # For large datasets, create alternative visualizations
                    print(f"Skipping confusion matrix visualization for {dataset_name} ({num_classes} classes - too large to visualize effectively)")
                    
                    # Plot class accuracy distribution instead
                    plt.figure(figsize=(12, 6))
                    plt.hist(class_accuracies, bins=50, alpha=0.7, edgecolor='black')
                    plt.xlabel('Per-class Accuracy')
                    plt.ylabel('Number of Classes')
                    plt.title(f'Distribution of Per-class Accuracies - {dataset_name}')
                    plt.axvline(class_accuracies.mean(), color='red', linestyle='--', label=f'Mean: {class_accuracies.mean():.3f}')
                    plt.legend()
                    wandb.log({f"{dataset_name}/class_accuracy_distribution": wandb.Image(plt)})
                    plt.close()
                
                # === NON-VISUAL LOGGING (for all datasets) ===
                
                # Log basic classification metrics
                wandb.log({
                    f"{dataset_name}/macro_f1": macro_f1,
                    f"{dataset_name}/accuracy": top1,
                    f"{dataset_name}/ece": mean_ece.item(),
                })
                
                # Log per-class metrics from classification report
                for class_idx in range(num_classes):
                    if str(class_idx) in class_report:
                        class_metrics = class_report[str(class_idx)]
                        wandb.log({
                            f"{dataset_name}/class_{class_idx}_precision": class_metrics.get('precision', 0),
                            f"{dataset_name}/class_{class_idx}_recall": class_metrics.get('recall', 0),
                            f"{dataset_name}/class_{class_idx}_f1": class_metrics.get('f1-score', 0),
                            f"{dataset_name}/class_{class_idx}_support": class_metrics.get('support', 0),
                        })
                
                # Log prediction distribution to detect class collapse (all datasets)
                pred_counts = np.bincount(y_pred, minlength=num_classes)
                true_counts = np.bincount(y_true, minlength=num_classes)
                
                for class_idx in range(num_classes):
                    wandb.log({
                        f"{dataset_name}/class_{class_idx}_pred_count": pred_counts[class_idx],
                        f"{dataset_name}/class_{class_idx}_true_count": true_counts[class_idx],
                    })
                
                # Log class performance summary statistics (all datasets)
                wandb.log({
                    f"{dataset_name}/class_acc_mean": class_accuracies.mean(),
                    f"{dataset_name}/class_acc_std": class_accuracies.std(),
                    f"{dataset_name}/class_acc_min": class_accuracies.min(),
                    f"{dataset_name}/class_acc_max": class_accuracies.max(),
                    f"{dataset_name}/classes_with_zero_acc": np.sum(class_accuracies == 0),
                    f"{dataset_name}/classes_with_perfect_acc": np.sum(class_accuracies == 1.0),
                    f"{dataset_name}/classes_with_low_acc": np.sum(class_accuracies < 0.1),  # <10% accuracy
                    f"{dataset_name}/classes_with_high_acc": np.sum(class_accuracies > 0.9),  # >90% accuracy
                })
                
                # Log worst and best performing classes (all datasets)
                worst_classes = np.argsort(class_accuracies)[:10]  # 10 worst classes
                best_classes = np.argsort(class_accuracies)[-10:]  # 10 best classes
                
                # Show worst and best performing classes in console for large datasets
                if num_classes > 50:
                    print(f"\n=== Top 10 Worst Performing Classes for {dataset_name} ===")
                    for i, class_idx in enumerate(worst_classes):
                        if class_totals[class_idx] > 0:
                            print(f"{i+1}. Class {class_idx}: {class_accuracies[class_idx]:.4f} ({class_correct[class_idx]}/{class_totals[class_idx]})")
                    
                    print(f"\n=== Top 10 Best Performing Classes for {dataset_name} ===")
                    for i, class_idx in enumerate(reversed(best_classes)):
                        if class_totals[class_idx] > 0:
                            print(f"{i+1}. Class {class_idx}: {class_accuracies[class_idx]:.4f} ({class_correct[class_idx]}/{class_totals[class_idx]})")
                
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")
            
    else:
        macro_f1 = float('nan')
        conf_matrix = None
        class_report = None

    if visualization:
        plot_dir = './plots'
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        visualize.draw_reliability_diagram(ys.cpu(), y_hats.cpu(), confidences.cpu(), num_bins=10, title=f'{args.model}', ece=mean_ece)
        file_name = f'{dataset_name}_{args.method}_ls{args.ls}_ts{args.temperature_scale}.png'
        plt.savefig(os.path.join(plot_dir, file_name))


    if hasattr(dataset, 'post_loop_metrics'):
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                            all_metadata, args)
        if 'acc' in metrics:
            metrics['top1'] = metrics['acc']
    else:
        metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    metrics['ece'] = mean_ece.item()
    metrics['macro_f1'] = macro_f1
    
    # Add detailed classification metrics to returned results
    if 'conf_matrix' in locals() and conf_matrix is not None:
        metrics['confusion_matrix'] = conf_matrix
        metrics['classification_report'] = class_report
        
        # Add class collapse detection metrics
        pred_counts = np.bincount(y_pred, minlength=num_classes)
        true_counts = np.bincount(y_true, minlength=num_classes)
        
        # Calculate prediction entropy to detect class collapse
        pred_probs = pred_counts / pred_counts.sum()
        pred_entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-12))
        max_entropy = np.log(num_classes)
        normalized_entropy = pred_entropy / max_entropy
        
        metrics['prediction_entropy'] = pred_entropy
        metrics['normalized_prediction_entropy'] = normalized_entropy
        metrics['num_predicted_classes'] = np.sum(pred_counts > 0)
        metrics['total_classes'] = num_classes
        
        print(f"\n=== Class Collapse Detection for {dataset_name} ===")
        print(f"Prediction entropy: {pred_entropy:.4f} (max: {max_entropy:.4f})")
        print(f"Normalized entropy: {normalized_entropy:.4f} (1.0 = uniform, 0.0 = collapsed)")
        print(f"Classes with predictions: {np.sum(pred_counts > 0)}/{num_classes}")
        
        # Warn about potential class collapse
        if normalized_entropy < 0.5:
            print(f"⚠️  WARNING: Low prediction entropy ({normalized_entropy:.4f}) suggests potential class collapse!")
        if np.sum(pred_counts > 0) < num_classes * 0.5:
            print(f"⚠️  WARNING: Only {np.sum(pred_counts > 0)} out of {num_classes} classes have predictions!")
    
    return metrics


def evaluate(
    image_classifier,
    args,
    classification_head=None,
    train_stats={},
    logger=None,
    bibim=False,
):
    if args.eval_datasets is None:
        return
    info = vars(args)

    for i, dataset_name in enumerate(args.eval_datasets):
        if bibim:
            if (i != 0) and (i != 4):
                continue

        print("Evaluating on", dataset_name)
        sub_dataset_locations = {} # {'dataset_desc' : location }

        dataset_class = getattr(datasets, dataset_name)

        dataset_desc = f"{dataset_name}"
        data_location = args.data_location
        sub_dataset_locations[dataset_desc] = data_location

        for dataset_desc, dataset_location in sub_dataset_locations.items():
            preprocess = image_classifier.val_preprocess if classification_head is None else image_classifier.module.val_preprocess

            flag = None; method = None
            dataset = dataset_class(preprocess,
                                    location=dataset_location,
                                    batch_size=args.batch_size,
                                    method=method,
                                    flag=flag)

            vis_flag = True if args.vis_calibration else False
            results = eval_single_dataset(image_classifier, dataset, args,
                    classification_head, visualization=vis_flag, dataset_name=dataset_name)
        
            if 'top1' in results:
                print(f"{dataset_desc} Top-1 accuracy: {results['top1']:.4f}")
                if logger != None:
                    logger.info(
                        f"{dataset_desc} Top-1 accuracy: {results['top1']:.4f}")
                train_stats[dataset_desc + " Accuracy"] = round(results['top1'], 4)
            
            if 'ece' in results:
                print(f"{dataset_desc} ECE: {results['ece']:.4f}")
                if logger != None:
                    logger.info(
                        f"{dataset_desc} ECE: {results['ece']:.4f}")
                train_stats[dataset_desc + " ECE"] = round(results['ece'], 4)

            if 'macro_f1' in results:
                print(f"{dataset_desc} Macro F1: {results['macro_f1']:.4f}")
                if logger != None:
                    logger.info(
                        f"{dataset_desc} Macro F1: {results['macro_f1']:.4f}")
                train_stats[dataset_desc + " Macro F1"] = round(results['macro_f1'], 4)
            
            # Log class collapse detection metrics
            if 'normalized_prediction_entropy' in results:
                entropy = results['normalized_prediction_entropy']
                num_pred_classes = results['num_predicted_classes']
                total_classes = results['total_classes']
                
                print(f"{dataset_desc} Prediction Entropy: {entropy:.4f}")
                print(f"{dataset_desc} Active Classes: {num_pred_classes}/{total_classes}")
                
                if logger != None:
                    logger.info(f"{dataset_desc} Prediction Entropy: {entropy:.4f}")
                    logger.info(f"{dataset_desc} Active Classes: {num_pred_classes}/{total_classes}")
                
                train_stats[dataset_desc + " Prediction Entropy"] = round(entropy, 4)
                train_stats[dataset_desc + " Active Classes"] = f"{num_pred_classes}/{total_classes}"
                
                # Log warning if class collapse detected
                if entropy < 0.5 or num_pred_classes < total_classes * 0.5:
                    warning_msg = f"⚠️  {dataset_desc}: Potential class collapse detected!"
                    print(warning_msg)
                    if logger != None:
                        logger.warning(warning_msg)

    return info
