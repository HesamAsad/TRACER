from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
from tqdm.auto import tqdm
import wandb
import pdb

import torch
from torch.nn import functional as F
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from src.datasets_.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits, clip_img_preprocessing, attack_pgd, apply_layer_freezing, cosine_grad_norm_scheduler
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.laion import get_data
from src.models.beta_moving_average import (
    GeneralMovingAverage,
    create_beta_weight_function,
    ExponentialMovingAverage,
    create_linear_warmup_ema_momentum,
)
import src.datasets_ as datasets


def lid_mom_est(data, reference, k, get_idx=False, 
                compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    Method of Moments estimation of Local Intrinsic Dimensionality (LID)
    using chordal distance for hyperspherical representations
    
    Args:
        data: representations that need LID to be estimated
        reference: reference representations (usually the same batch)
        k: locality parameter, the neighbourhood size
        get_idx: whether to return indices of nearest neighbors
        compute_mode: computation mode for cdist (kept for compatibility)
    
    Returns:
        lids: estimated LID values for each sample
    """
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    
    # Normalize vectors to unit length for chordal distance
    # (Note: CLIP features are already normalized, but this ensures robustness)
    data_norm = torch.nn.functional.normalize(data, p=2, dim=1)
    reference_norm = torch.nn.functional.normalize(reference, p=2, dim=1)
    
    # Compute cosine similarity matrix
    cosine_sim = torch.mm(data_norm, reference_norm.T)
    
    # Compute chordal distance: sqrt(2 - 2x^T y)
    # Stabilize by clamping the argument to avoid numerical issues
    chord_arg = 2.0 - 2.0 * cosine_sim
    chord_arg = torch.clamp(chord_arg, min=1e-8)  # Prevent sqrt of negative values
    r = torch.sqrt(chord_arg)
    
    # Sort distances and get k nearest neighbors
    a, idx = torch.sort(r, dim=1)
    
    # Compute mean distance to k nearest neighbors (excluding self)
    m = torch.mean(a[:, 1:k+1], dim=1)
    
    # Estimate LID using method of moments
    # Handle numerical issues: if denominator is too small, use a small positive value
    denominator = a[:, k+1] - m
    denominator = torch.clamp(denominator, min=1e-8)  # Prevent division by zero
    lids = m / denominator
    
    # Handle potential numerical issues
    lids = torch.clamp(lids, min=1e-8, max=1e8)  # Prevent both NaN and extreme values
    
    if get_idx:
        return idx, lids
    return lids


def compute_ldreg_loss(image_features, text_features, k=64, reg_type="l1"):
    """
    Compute LID regularization loss for both image and text modalities
    using chordal distance and sum their regularization losses
    
    Args:
        image_features: image representation features
        text_features: text representation features
        k: number of nearest neighbors
        reg_type: type of regularization ("l1" or "l2")
    
    Returns:
        ldreg_loss: combined LID regularization loss
        mean_lid_image: mean LID value for image features (for logging)
        mean_lid_text: mean LID value for text features (for logging)
    """
    # Estimate LID for image features
    lids_image = lid_mom_est(data=image_features, reference=image_features.detach(), k=k)
    
    # Estimate LID for text features
    lids_text = lid_mom_est(data=text_features, reference=text_features.detach(), k=k)
    
    # Compute regularization loss based on type
    if reg_type == "l1":
        ldreg_loss_image = -torch.log(lids_image).mean()
        ldreg_loss_text = -torch.log(lids_text).mean()
    elif reg_type == "l2":
        ldreg_loss_image = -torch.sqrt(torch.square(torch.log(lids_image))).mean()
        ldreg_loss_text = -torch.sqrt(torch.square(torch.log(lids_text))).mean()
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    
    # Sum the regularization losses
    ldreg_loss = ldreg_loss_image + ldreg_loss_text
    
    # Compute geometric mean of LID for logging
    mean_lid_image = torch.exp(torch.log(lids_image).mean())
    mean_lid_text = torch.exp(torch.log(lids_text).mean())
    
    return ldreg_loss, mean_lid_image.item(), mean_lid_text.item()


def carot_ldreg_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."

    logger.info("Fine-tuning Using CaRot Loss with LDReg")
    model = clip_encoder

    # Apply layer freezing based on arguments
    apply_layer_freezing(model, args, logger)

    input_key = "images"
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 5

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    dataset = dataset_class(
        preprocess_fn, location=args.data_location, batch_size=args.batch_size
    )

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0
    )
    assert len(img_text_data), "At least one train or eval dataset must be specified."
    ft_dataloader = img_text_data["train_ft"].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler('cuda')

    if args.clip_load is not None:
        model = model.load(args.clip_load)

    if args.distil_coef:
        # Create teacher encoder: EMA (if enabled) or Beta-MA (default)
        total_iterations = args.epochs * num_batches
        teacher_model = model.cuda()
        if getattr(args, 'ema_teacher', False):
            get_momentum_fn = create_linear_warmup_ema_momentum(
                src_momentum=args.m_sche_src,
                tar_momentum=args.m_sche_tar,
                warmup_ratio=args.m_warm_up,
                total_iterations=total_iterations,
            )
            ema_up_freq = args.ema_up_freq if args.ema_up_freq > 0 else 500
            teacher_enc = ExponentialMovingAverage(teacher_model, get_momentum_fn, update_frequency=ema_up_freq)
        else:
            weight_func = create_beta_weight_function(args.beta, total_iterations)
            teacher_enc = GeneralMovingAverage(teacher_model, weight_func)

        ema_up_freq = args.ema_up_freq if not getattr(args, 'ema_teacher', False) else (args.ema_up_freq if args.ema_up_freq > 0 else 500)

    model = model.cuda()

    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info("Using devices" + str(devices))

    model = torch.nn.DataParallel(model, device_ids=devices)

    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)
    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
        ls=args.ls,
    )

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    print(f"Number of trainable parameters: {len(params)}")
    logger.info(f"Number of trainable parameters: {len(params)}")
    wandb.log({"trainable params": len(params)})
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer, args.lr, args.warmup_length, args.epochs * num_batches, args.min_lr
    )

    # Initialize gradient norm scheduler
    initial_grad_norm = args.max_grad_norm  # Start from initial value (0.0001)
    final_grad_norm = args.max_grad_norm * args.grad_norm_multiplier  # Target final value
    grad_norm_scheduler = cosine_grad_norm_scheduler(
        initial_grad_norm, final_grad_norm, args.epochs * num_batches
    )

    stats = []
    prev_num_logits = 0
    labels_ = {}
    #! inference flag
    if args.epochs == 0:
        epoch = 0
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        args.current_epoch = epoch
        
        print("Start evaluation")
        classification_head_new = get_zeroshot_classifier(args, model.module.model)
        classification_head_new = classification_head_new.cuda()
        eval_results = evaluate(
            model, args, classification_head_new, epoch_stats, logger
        )
        wandb.log({k: v for k, v in epoch_stats.items()})
        exit()

    for epoch in tqdm(range(0, args.epochs), desc="Epochs"):
        print("\nEpoch : ", epoch)
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        # Initialize tracking variables for epoch statistics
        id_carot_loss_sum = 0
        ldreg_loss_sum = 0
        mean_lid_image_sum = 0
        mean_lid_text_sum = 0
        fnorm_loss_sum = 0
        orth_loss_sum = 0
        dist_loss_sum = 0
        clip_loss_sum = 0
        supcon_logged_this_epoch = False  # Track if we've logged supervised contrastive info this epoch
        model.train()
        model = model.cuda()
        classification_head.train()

        for i in tqdm(range(num_batches), desc="Batches"):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)

            # Update gradient norm for this step
            current_grad_norm = grad_norm_scheduler(step)

            optimizer.zero_grad()

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ft_iterator = iter(
                    ft_dataloader
                )  # If ft_iterator is all used, re-initialize it
                ft_batch = next(ft_iterator)
            
            # Try to unpack labels if available
            ft_labels = None
            use_supcon = False
            if len(ft_batch) == 3:
                ft_image, ft_text, ft_labels = ft_batch
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()
                ft_labels = ft_labels.cuda()
                use_supcon = True
                if not supcon_logged_this_epoch:
                    logger.info(f"Using supervised CLIP loss with labels for epoch {epoch}")
                    supcon_logged_this_epoch = True
            else:
                ft_image, ft_text = ft_batch
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32):
                ft_image_features, ft_text_features, logit_scale2 = model(
                    ft_image, ft_text
                )

                lscale = logit_scale2 if len(devices) == 1 else logit_scale2[0]

                ft_clip_loss, logits_per_image, logits_per_text = clip_loss_fn(
                    ft_image_features, ft_text_features, lscale
                )

                #* d-rank SVD approximation
                if args.cross_fnorm:
                    if args.model[:3] != 'ViT':
                        cov_vl = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.text_projection.T
                    else:
                        cov_vl = model.module.model.visual.proj @ model.module.model.text_projection.T
                    fnorm_val = torch.linalg.norm(cov_vl, ord='fro')
                    ft_clip_loss += args.cross_fnorm * fnorm_val
                    fnorm_val = fnorm_val.item()
                    fnorm_loss_sum += args.cross_fnorm * fnorm_val

                #* orthogonality constraint
                if args.l_orth_wv:
                    if args.model[:3] != 'ViT':
                        covv = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.visual.attnpool.c_proj
                    else:
                        covv = model.module.model.visual.proj.T @ model.module.model.visual.proj
                    orth_val = ((covv - torch.eye(covv.shape[0], device=covv.device))**2).sum()**(1/2)
                    ft_clip_loss += args.l_orth_wv * orth_val
                    orth_val = orth_val.item()
                    orth_loss_sum += args.l_orth_wv * orth_val

                #! LDReg regularization
                ldreg_loss, mean_lid_image, mean_lid_text = compute_ldreg_loss(
                    ft_image_features, ft_text_features, 
                    k=args.ldreg_k, 
                    reg_type=args.ldreg_type
                )
                ldreg_loss_val = ldreg_loss.item()
                ft_clip_loss += args.ldreg_coef * ldreg_loss
                ldreg_loss_sum += args.ldreg_coef * ldreg_loss_val
                mean_lid_image_sum += mean_lid_image
                mean_lid_text_sum += mean_lid_text

            #! self-distillation flag
            dist_loss, current_weight = torch.tensor(0), 0.0
            if args.distil_coef:
                if step > 0:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32):
                        with torch.no_grad():
                            # Use the Beta moving average teacher for inference
                            (
                                ft_image_features_t,
                                ft_text_features_t,
                                logit_scale_t,
                            ) = teacher_enc.moving_avg(ft_image, ft_text)

                            logits_per_image_t = (
                                logit_scale_t
                                * ft_image_features_t
                                @ ft_text_features_t.T
                            )
                            logits_per_text_t = (
                                logit_scale_t
                                * ft_text_features_t
                                @ ft_image_features_t.T
                            )
                        
                        dist_loss = -torch.sum(
                            F.softmax(logits_per_image_t, dim=1)
                            * torch.log(F.softmax(logits_per_image, dim=1))
                            + F.softmax(logits_per_text_t, dim=1)
                            * torch.log(F.softmax(logits_per_text, dim=1)),
                            dim=1
                        ).mean()
                        
                        ft_clip_loss += args.distil_coef * dist_loss
                        if isinstance(dist_loss, torch.Tensor):
                            dist_loss_sum += args.distil_coef * dist_loss.item()
                        
                        # Get current momentum/weight for logging
                        current_weight = getattr(teacher_enc, 'weight', getattr(teacher_enc, 'momentum', 0.0))

            if fp16_scaler is None:
                ft_clip_loss.backward()
                # Apply gradient clipping if specified
                grad_norm = None
                if current_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, current_grad_norm)

                optimizer.step()
            else:
                fp16_scaler.scale(ft_clip_loss).backward()
                # Apply gradient clipping if specified
                grad_norm = None
                if current_grad_norm > 0:
                    fp16_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, current_grad_norm)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            #! self-distillation
            if args.distil_coef:
                total_steps = num_batches * args.epochs
                if getattr(args, 'ema_teacher', False):
                    teacher_update_freq = args.ema_up_freq if args.ema_up_freq > 0 else 500
                    if teacher_update_freq > 0 and (((step % teacher_update_freq) == 0) or (step == total_steps)):
                        teacher_enc.update(global_step=step)
                else:
                    if args.ema_up_freq <= 0:
                        teacher_enc.update()
                    else:
                        if ((step % args.ema_up_freq) == 0) or (step == total_steps - 1):
                            teacher_enc.update()

            # Track base CLIP loss
            base_clip_loss = ft_clip_loss.item()
            if args.cross_fnorm:
                base_clip_loss -= args.cross_fnorm * fnorm_val
            if args.l_orth_wv:
                base_clip_loss -= args.l_orth_wv * orth_val
            if args.ldreg_coef > 0:
                base_clip_loss -= args.ldreg_coef * ldreg_loss_val
            if args.distil_coef and isinstance(dist_loss, torch.Tensor):
                base_clip_loss -= args.distil_coef * dist_loss.item()
            clip_loss_sum += base_clip_loss

            id_carot_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                
                # Prepare detailed log message
                log_msg = (
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\n"
                    f"\tTotal Loss: {ft_clip_loss.item():.4f}\n"
                    f"\tCLIP Loss: {base_clip_loss:.4f}"
                )
                
                # Prepare wandb log dict
                wandb_log = {
                    "Train Epoch": epoch,
                    "Percent Complete": percent_complete,
                    "Total Loss": ft_clip_loss.item(),
                    "CLIP Loss": base_clip_loss,
                }
                
                # Add gradient norm if clipping is enabled
                if current_grad_norm > 0 and grad_norm is not None:
                    log_msg += f"\n\tGradient Norm: {grad_norm:.4f} (scheduled max: {current_grad_norm:.6f})"
                    wandb_log.update({
                        "Gradient Norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "Scheduled Max Grad Norm": current_grad_norm,
                        "Initial Max Grad Norm": initial_grad_norm,
                        "Final Max Grad Norm": final_grad_norm,
                    })
                
                # Add cross_fnorm loss if applicable
                if args.cross_fnorm:
                    log_msg += f"\n\tCross F-norm Loss: {args.cross_fnorm * fnorm_val:.4f} (F-norm: {fnorm_val:.4f})"
                    wandb_log.update({
                        "Cross F-norm Loss": args.cross_fnorm * fnorm_val,
                        "F-norm Value": fnorm_val,
                    })
                
                # Add orthogonality loss if applicable
                if args.l_orth_wv:
                    log_msg += f"\n\tOrthogonality Loss: {args.l_orth_wv * orth_val:.4f} (Orth: {orth_val:.4f})"
                    wandb_log.update({
                        "Orthogonality Loss": args.l_orth_wv * orth_val,
                        "Orthogonality Value": orth_val,
                    })
                
                # Add LDReg loss if applicable
                if args.ldreg_coef > 0:
                    log_msg += f"\n\tLDReg Loss: {args.ldreg_coef * ldreg_loss_val:.4f} (Raw: {ldreg_loss_val:.4f})"
                    log_msg += f"\n\tMean LID Image: {mean_lid_image:.2f}"
                    log_msg += f"\n\tMean LID Text: {mean_lid_text:.2f}"
                    wandb_log.update({
                        "LDReg Loss": args.ldreg_coef * ldreg_loss_val,
                        "LDReg Raw Loss": ldreg_loss_val,
                        "Mean LID Image": mean_lid_image,
                        "Mean LID Text": mean_lid_text,
                    })
                
                # Add distillation loss if applicable
                if args.distil_coef and isinstance(dist_loss, torch.Tensor):
                    # Calculate training progress for beta momentum display
                    total_steps = num_batches * args.epochs
                    progress = step / total_steps
                    log_msg += f"\n\tDistillation Loss: {args.distil_coef * dist_loss.item():.4f} (Raw: {dist_loss.item():.4f})"
                    momentum_label = "EMA Momentum" if getattr(args, 'ema_teacher', False) else "Beta Momentum"
                    log_msg += f"\n\t{momentum_label}: {current_weight:.4f} (progress: {progress:.2%})"
                    wandb_log.update({
                        "Distillation Loss": args.distil_coef * dist_loss.item(),
                        "Distillation Raw Loss": dist_loss.item(),
                        (momentum_label): current_weight,
                        "Training Progress": progress,
                    })
                
                # Add learning rate
                current_lr = optimizer.param_groups[0]['lr']
                log_msg += f"\n\tLearning Rate: {current_lr:.6f}"
                wandb_log.update({"Learning Rate": current_lr})
                
                # Add logit scale
                log_msg += f"\n\tLogit Scale: {lscale.exp().item():.4f}"
                wandb_log.update({"Logit Scale": lscale.exp().item()})
                
                logger.info(log_msg)
                wandb.log(wandb_log)

        # Compute averages
        id_carot_loss_avg = id_carot_loss_sum / num_batches
        clip_loss_avg = clip_loss_sum / num_batches

        epoch_stats["Avg Total Loss"] = round(id_carot_loss_avg, 4)
        epoch_stats["Avg CLIP Loss"] = round(clip_loss_avg, 4)

        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Avg Total Loss: {id_carot_loss_avg:.4f}")
        logger.info(f"  Avg CLIP Loss: {clip_loss_avg:.4f}")

        if args.cross_fnorm:
            fnorm_loss_avg = fnorm_loss_sum / num_batches
            epoch_stats["Avg Cross F-norm Loss"] = round(fnorm_loss_avg, 4)
            logger.info(f"  Avg Cross F-norm Loss: {fnorm_loss_avg:.4f}")

        if args.l_orth_wv:
            orth_loss_avg = orth_loss_sum / num_batches
            epoch_stats["Avg Orthogonality Loss"] = round(orth_loss_avg, 4)
            logger.info(f"  Avg Orthogonality Loss: {orth_loss_avg:.4f}")

        if args.ldreg_coef > 0:
            ldreg_loss_avg = ldreg_loss_sum / num_batches
            mean_lid_image_avg = mean_lid_image_sum / num_batches
            mean_lid_text_avg = mean_lid_text_sum / num_batches
            epoch_stats["Avg LDReg Loss"] = round(ldreg_loss_avg, 4)
            epoch_stats["Avg Mean LID Image"] = round(mean_lid_image_avg, 2)
            epoch_stats["Avg Mean LID Text"] = round(mean_lid_text_avg, 2)
            logger.info(f"  Avg LDReg Loss: {ldreg_loss_avg:.4f}")
            logger.info(f"  Avg Mean LID Image: {mean_lid_image_avg:.2f}")
            logger.info(f"  Avg Mean LID Text: {mean_lid_text_avg:.2f}")

        if args.distil_coef:
            dist_loss_avg = dist_loss_sum / num_batches
            epoch_stats["Avg Distillation Loss"] = round(dist_loss_avg, 4)
            logger.info(f"  Avg Distillation Loss: {dist_loss_avg:.4f}")

        # Log final learning rate for the epoch
        final_lr = optimizer.param_groups[0]['lr']
        epoch_stats["Final LR"] = final_lr
        logger.info(f"  Final Learning Rate: {final_lr:.6f}")

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f"checkpoint_{epoch+1}.pt")
            logger.info("Saving model to" + str(model_path))
            model.module.save(model_path)

            #! save the EMA teacher
            if args.distil_coef:
                ema_model_path = os.path.join(args.save, f"checkpoint_{epoch+1}_EMA.pt")
                logger.info("Saving model to" + str(ema_model_path))
                try:
                    teacher_enc.save(ema_model_path)
                except:
                    print("============================")
                    print("error occurred during EMA model saving")
                    print("============================")

            optim_path = os.path.join(args.save, f"optim_{epoch+1}.pt")
            torch.save(optimizer.state_dict(), optim_path)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32), torch.no_grad():
            evaluate(model, args, classification_head_new, epoch_stats, logger)
        
        ood_acc = 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if "Accuracy" in k:
                if k == "ImageNet Accuracy":
                    # ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0

        epoch_stats["Avg OOD Acc"] = round(ood_acc, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        
        # Define model flag for more descriptive log directory
        mod_flag = args.model.split('/')[-1] if '/' in args.model else args.model
        
        log_dir = (
            "expt_logs/"
            + args.exp_name
            + "/"
            + f"{mod_flag}_ep{args.epochs}"
            + f"_BS{args.batch_size}"
            + f"_WD{args.wd}"
            + f"_LR{args.lr}"
            + f"_D{args.distil_coef}"
            + f"_OC{args.l_orth_wv}"
            + f"_CF{args.cross_fnorm}"
            + f"_LDReg{args.ldreg_coef}"
            + f"_k{args.ldreg_k}"
            + f"_run{args.run}"
        )
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + "/stats.tsv", sep="\t")

        #! wandb logging
        wandb.log({k: v for k, v in epoch_stats.items()})

    if args.save is not None:
        return model_path
