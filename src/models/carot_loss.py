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
from src.models.utils import cosine_lr, cosine_grad_norm_scheduler, apply_layer_freezing, torch_load, LabelSmoothing, get_logits, clip_img_preprocessing, attack_pgd
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.laion import get_data
from src.models.beta_moving_average import GeneralMovingAverage, create_beta_weight_function
from src.models.gradient_diagnostics import GradientDiagnostics, create_loss_dict_for_diagnostics
import src.datasets_ as datasets


def carot_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."

    logger.info("Fine-tuning Using carot Loss")
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
        total_iterations = args.epochs * num_batches
        
        if args.dual_teacher:
            # Dynamic Dual-Teacher Distillation Setup
            logger.info("Setting up Dynamic Dual-Teacher Distillation")
            logger.info(f"ID Teacher (EMA) update frequency: {args.ema_up_freq} ({'every step' if args.ema_up_freq <= 0 else f'every {args.ema_up_freq} steps'})")
            logger.info(f"OOD Teacher (BMA) update frequency: {args.bma_up_freq} ({'every step' if args.bma_up_freq <= 0 else f'every {args.bma_up_freq} steps'})")
            
            # ID Teacher (EMA) - Expert on in-distribution data
            if args.use_old_ema:
                id_teacher_enc = GeneralMovingAverage(
                    model.cuda(), 
                    weight_func=None,
                    use_old=True,
                    m_sche_src=args.m_sche_src,
                    m_sche_tar=args.m_sche_tar,
                    m_warm_up=args.m_warm_up,
                    total_steps=total_iterations,
                    ema_up_freq=args.ema_up_freq
                )
            else:
                # Use standard EMA for ID teacher (momentum = 0.999)
                ema_weight_func = lambda x: 0.999  # Constant momentum for EMA
                id_teacher_enc = GeneralMovingAverage(model.cuda(), ema_weight_func)
            
            # OOD Teacher (BMA) - Better at out-of-distribution generalization
            bma_weight_func = create_beta_weight_function(args.beta, total_iterations)
            ood_teacher_enc = GeneralMovingAverage(model.cuda(), bma_weight_func)
            
            logger.info(f"Dual-Teacher Setup Complete: ID Teacher (EMA), OOD Teacher (BMA), entropy_threshold={args.entropy_threshold}")
            
        elif args.use_old_ema:
            # Single teacher - old momentum-based EMA implementation
            teacher_enc = GeneralMovingAverage(
                model.cuda(), 
                weight_func=None,
                use_old=True,
                m_sche_src=args.m_sche_src,
                m_sche_tar=args.m_sche_tar,
                m_warm_up=args.m_warm_up,
                total_steps=total_iterations,
                ema_up_freq=args.ema_up_freq
            )
        else:
            # Single teacher - Beta distribution-based moving average
            weight_func = create_beta_weight_function(args.beta, total_iterations)
            teacher_enc = GeneralMovingAverage(model.cuda(), weight_func)

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
    
    # Initialize gradient diagnostics if enabled
    grad_diagnostics = None
    if args.enable_grad_diagnostics:
        logger.info("Initializing gradient diagnostics system")
        grad_diagnostics = GradientDiagnostics(
            model=model.module if isinstance(model, torch.nn.DataParallel) else model,
            args=args,
            logger=logger,
            log_frequency=print_every
        )
        grad_diagnostics.update_total_steps(args.epochs * num_batches)

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
        fnorm_loss_sum = 0
        orth_loss_sum = 0
        dist_loss_sum = 0
        clip_loss_sum = 0
        supcon_logged_this_epoch = False  # Track if we've logged supervised contrastive info this epoch
        
        # Dual-teacher epoch statistics
        epoch_dual_teacher_stats = {
            'total_id_selections': 0,
            'total_ood_selections': 0,
            'total_agreements': 0,
            'sum_entropy_id': 0.0,
            'sum_entropy_ood': 0.0,
            'sum_entropy_diff': 0.0,
            'batch_count': 0,
            'sum_id_confidence': 0.0,
            'sum_ood_confidence': 0.0,
            'sum_both_confident': 0.0,
            'sum_agreement_rate': 0.0,
            'id_teacher_updates': 0,
            'ood_teacher_updates': 0,
            'total_steps': 0
        }
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

                # Initialize individual loss components for gradient diagnostics
                base_clip_loss_tensor = ft_clip_loss
                crossf_loss_raw = None
                oc_loss_raw = None
                
                #* d-rank SVD approximation
                if args.cross_fnorm:
                    if args.model[:3] != 'ViT':
                        cov_vl = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.text_projection.T
                    else:
                        cov_vl = model.module.model.visual.proj @ model.module.model.text_projection.T
                    crossf_loss_raw = torch.linalg.norm(cov_vl, ord='fro')
                    ft_clip_loss += args.cross_fnorm * crossf_loss_raw
                    fnorm_val = crossf_loss_raw.item()
                    fnorm_loss_sum += args.cross_fnorm * fnorm_val

                #* orthogonality constraint
                if args.l_orth_wv:
                    if args.model[:3] != 'ViT':
                        covv = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.visual.attnpool.c_proj
                    else:
                        covv = model.module.model.visual.proj.T @ model.module.model.visual.proj
                    oc_loss_raw = ((covv - torch.eye(covv.shape[0], device=covv.device))**2).sum()**(1/2)
                    ft_clip_loss += args.l_orth_wv * oc_loss_raw
                    orth_val = oc_loss_raw.item()
                    orth_loss_sum += args.l_orth_wv * orth_val

            #! self-distillation flag
            dist_loss_raw, current_weight = torch.tensor(0.0).cuda(), 0.0
            teacher_selection_stats = {
                'id_selected': 0, 
                'ood_selected': 0, 
                'agreement_override': 0,
                'total_samples': 0,
                'id_confidence_rate': 0.0,
                'ood_confidence_rate': 0.0,
                'both_confident_rate': 0.0,
                'prediction_agreement_rate': 0.0,
                # New correctness-based metrics
                'id_correct_rate': 0.0,
                'ood_correct_rate': 0.0,
                'correctness_override_count': 0,
                'id_selected_by_correctness': 0,
                'ood_selected_by_correctness': 0,
                'id_selected_by_confidence': 0,
                'ood_selected_by_confidence': 0,
                'both_correct_rate': 0.0,
                'both_wrong_rate': 0.0,
                'correctness_disagree_rate': 0.0
            }
            entropy_stats = {'mean_entropy_id': 0.0, 'mean_entropy_ood': 0.0, 'entropy_diff': 0.0}
            teacher_update_stats = {'id_updated': False, 'ood_updated': False}
            
            if args.distil_coef:
                if step > 0:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32):
                        with torch.no_grad():
                            if args.dual_teacher:
                                # Dynamic Dual-Teacher Distillation
                                
                                # Get predictions from both teachers
                                (
                                    ft_image_features_id,
                                    ft_text_features_id,
                                    logit_scale_id,
                                ) = id_teacher_enc.moving_avg(ft_image, ft_text)
                                
                                (
                                    ft_image_features_ood,
                                    ft_text_features_ood,
                                    logit_scale_ood,
                                ) = ood_teacher_enc.moving_avg(ft_image, ft_text)

                                logits_per_image_id = (
                                    logit_scale_id
                                    * ft_image_features_id
                                    @ ft_text_features_id.T
                                )
                                logits_per_text_id = (
                                    logit_scale_id
                                    * ft_text_features_id
                                    @ ft_image_features_id.T
                                )
                                
                                logits_per_image_ood = (
                                    logit_scale_ood
                                    * ft_image_features_ood
                                    @ ft_text_features_ood.T
                                )
                                logits_per_text_ood = (
                                    logit_scale_ood
                                    * ft_text_features_ood
                                    @ ft_image_features_ood.T
                                )

                                # Calculate probabilities and entropies for dynamic selection
                                probs_id_img = F.softmax(logits_per_image_id, dim=1)
                                probs_ood_img = F.softmax(logits_per_image_ood, dim=1)
                                entropy_id_img = -torch.sum(probs_id_img * torch.log(probs_id_img + 1e-8), dim=1)
                                entropy_ood_img = -torch.sum(probs_ood_img * torch.log(probs_ood_img + 1e-8), dim=1)
                                
                                probs_id_txt = F.softmax(logits_per_text_id, dim=1)
                                probs_ood_txt = F.softmax(logits_per_text_ood, dim=1)
                                entropy_id_txt = -torch.sum(probs_id_txt * torch.log(probs_id_txt + 1e-8), dim=1)
                                entropy_ood_txt = -torch.sum(probs_ood_txt * torch.log(probs_ood_txt + 1e-8), dim=1)

                                # Dynamic Teacher Selection Logic with Correctness Check
                                # 
                                # NEW APPROACH: Correctness-Aware Teacher Selection
                                # ================================================
                                # Rule 1: Correctness trumps confidence - always prefer correct teacher over incorrect one
                                # Rule 2: If both correct or both wrong, use confidence (lower entropy = more confident)
                                # Rule 3: If both confident and agree, prefer ID teacher (for in-distribution performance)
                                #
                                # This ensures we don't follow confident but wrong teachers, which was a major flaw
                                # in the previous entropy-only approach.
                                #
                                # Ground truth for CLIP: each image should match with its corresponding text (diagonal)
                                batch_size = ft_image.size(0)
                                ground_truth = torch.arange(batch_size, device=ft_image.device)
                                
                                # Get predictions from both teachers
                                preds_id_img = torch.argmax(probs_id_img, dim=1)
                                preds_ood_img = torch.argmax(probs_ood_img, dim=1)
                                preds_id_txt = torch.argmax(probs_id_txt, dim=1)
                                preds_ood_txt = torch.argmax(probs_ood_txt, dim=1)
                                
                                # Check correctness for each teacher
                                id_correct_img = (preds_id_img == ground_truth)
                                ood_correct_img = (preds_ood_img == ground_truth)
                                id_correct_txt = (preds_id_txt == ground_truth)
                                ood_correct_txt = (preds_ood_txt == ground_truth)
                                
                                # Initialize target logits with OOD teacher (default)
                                target_logits_img = logits_per_image_ood.clone()
                                target_logits_txt = logits_per_text_ood.clone()
                                
                                # Rule 1: Correctness trumps confidence - use correct teacher over incorrect one
                                # For images: if ID correct and OOD wrong, use ID; if OOD correct and ID wrong, use OOD
                                id_correct_ood_wrong_img = id_correct_img & (~ood_correct_img)
                                ood_correct_id_wrong_img = ood_correct_img & (~id_correct_img)
                                
                                target_logits_img[id_correct_ood_wrong_img] = logits_per_image_id[id_correct_ood_wrong_img]
                                # target_logits_img already has OOD for ood_correct_id_wrong_img case
                                
                                # For text: same logic
                                id_correct_ood_wrong_txt = id_correct_txt & (~ood_correct_txt)
                                ood_correct_id_wrong_txt = ood_correct_txt & (~id_correct_txt)
                                
                                target_logits_txt[id_correct_ood_wrong_txt] = logits_per_text_id[id_correct_ood_wrong_txt]
                                # target_logits_txt already has OOD for ood_correct_id_wrong_txt case
                                
                                # Rule 2: If both correct or both wrong, use confidence (lower entropy = more confident)
                                both_correct_or_both_wrong_img = ((id_correct_img & ood_correct_img) | ((~id_correct_img) & (~ood_correct_img)))
                                both_correct_or_both_wrong_txt = ((id_correct_txt & ood_correct_txt) | ((~id_correct_txt) & (~ood_correct_txt)))
                                
                                # Among samples where both are correct/wrong, choose more confident one
                                id_more_confident_img = (entropy_id_img < entropy_ood_img)
                                id_more_confident_txt = (entropy_id_txt < entropy_ood_txt)
                                
                                use_id_for_confidence_img = both_correct_or_both_wrong_img & id_more_confident_img
                                use_id_for_confidence_txt = both_correct_or_both_wrong_txt & id_more_confident_txt
                                
                                target_logits_img[use_id_for_confidence_img] = logits_per_image_id[use_id_for_confidence_img]
                                target_logits_txt[use_id_for_confidence_txt] = logits_per_text_id[use_id_for_confidence_txt]
                                
                                # Rule 3: Override - If both teachers are confident and agree, prefer ID teacher (for ID performance)
                                confident_and_agree_img = (
                                    (entropy_id_img < args.entropy_threshold) & 
                                    (entropy_ood_img < args.entropy_threshold) & 
                                    (preds_id_img == preds_ood_img)
                                )
                                confident_and_agree_txt = (
                                    (entropy_id_txt < args.entropy_threshold) & 
                                    (entropy_ood_txt < args.entropy_threshold) & 
                                    (preds_id_txt == preds_ood_txt)
                                )
                                
                                target_logits_img[confident_and_agree_img] = logits_per_image_id[confident_and_agree_img]
                                target_logits_txt[confident_and_agree_txt] = logits_per_text_id[confident_and_agree_txt]
                                
                                # Calculate entropy statistics
                                mean_entropy_id = (entropy_id_img.mean() + entropy_id_txt.mean()) / 2
                                mean_entropy_ood = (entropy_ood_img.mean() + entropy_ood_txt.mean()) / 2
                                entropy_diff = mean_entropy_id - mean_entropy_ood
                                
                                entropy_stats['mean_entropy_id'] = mean_entropy_id.item()
                                entropy_stats['mean_entropy_ood'] = mean_entropy_ood.item()
                                entropy_stats['entropy_diff'] = entropy_diff.item()
                                
                                # Enhanced teacher selection statistics with correctness tracking
                                batch_size = ft_image.size(0)
                                teacher_selection_stats['total_samples'] = batch_size * 2  # img + txt
                                
                                # Correctness rates
                                teacher_selection_stats['id_correct_rate'] = (
                                    id_correct_img.sum() + id_correct_txt.sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['ood_correct_rate'] = (
                                    ood_correct_img.sum() + ood_correct_txt.sum()
                                ).item() / (batch_size * 2)
                                
                                # Correctness agreement/disagreement rates
                                teacher_selection_stats['both_correct_rate'] = (
                                    (id_correct_img & ood_correct_img).sum() + (id_correct_txt & ood_correct_txt).sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['both_wrong_rate'] = (
                                    ((~id_correct_img) & (~ood_correct_img)).sum() + ((~id_correct_txt) & (~ood_correct_txt)).sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['correctness_disagree_rate'] = (
                                    (id_correct_img ^ ood_correct_img).sum() + (id_correct_txt ^ ood_correct_txt).sum()
                                ).item() / (batch_size * 2)
                                
                                # Selection by correctness (Rule 1)
                                teacher_selection_stats['id_selected_by_correctness'] = (
                                    id_correct_ood_wrong_img.sum() + id_correct_ood_wrong_txt.sum()
                                ).item()
                                teacher_selection_stats['ood_selected_by_correctness'] = (
                                    ood_correct_id_wrong_img.sum() + ood_correct_id_wrong_txt.sum()
                                ).item()
                                teacher_selection_stats['correctness_override_count'] = (
                                    teacher_selection_stats['id_selected_by_correctness'] + 
                                    teacher_selection_stats['ood_selected_by_correctness']
                                )
                                
                                # Selection by confidence (Rule 2)
                                teacher_selection_stats['id_selected_by_confidence'] = (
                                    use_id_for_confidence_img.sum() + use_id_for_confidence_txt.sum()
                                ).item()
                                teacher_selection_stats['ood_selected_by_confidence'] = (
                                    (both_correct_or_both_wrong_img & (~id_more_confident_img)).sum() + 
                                    (both_correct_or_both_wrong_txt & (~id_more_confident_txt)).sum()
                                ).item()
                                
                                # Total selections (for backwards compatibility with existing logging)
                                teacher_selection_stats['id_selected'] = (
                                    teacher_selection_stats['id_selected_by_correctness'] + 
                                    teacher_selection_stats['id_selected_by_confidence']
                                )
                                teacher_selection_stats['ood_selected'] = (
                                    teacher_selection_stats['ood_selected_by_correctness'] + 
                                    teacher_selection_stats['ood_selected_by_confidence']
                                )
                                teacher_selection_stats['agreement_override'] = (confident_and_agree_img.sum() + confident_and_agree_txt.sum()).item()
                                
                                # Confidence and agreement metrics (existing)
                                teacher_selection_stats['id_confidence_rate'] = (
                                    (entropy_id_img < args.entropy_threshold).sum() + 
                                    (entropy_id_txt < args.entropy_threshold).sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['ood_confidence_rate'] = (
                                    (entropy_ood_img < args.entropy_threshold).sum() + 
                                    (entropy_ood_txt < args.entropy_threshold).sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['both_confident_rate'] = (
                                    ((entropy_id_img < args.entropy_threshold) & (entropy_ood_img < args.entropy_threshold)).sum() +
                                    ((entropy_id_txt < args.entropy_threshold) & (entropy_ood_txt < args.entropy_threshold)).sum()
                                ).item() / (batch_size * 2)
                                teacher_selection_stats['prediction_agreement_rate'] = (
                                    (preds_id_img == preds_ood_img).sum() + (preds_id_txt == preds_ood_txt).sum()
                                ).item() / (batch_size * 2)
                                
                                # Calculate distillation loss with dynamically selected targets
                                dist_loss_raw = -torch.sum(
                                    F.softmax(target_logits_img, dim=1)
                                    * torch.log(F.softmax(logits_per_image, dim=1))
                                    + F.softmax(target_logits_txt, dim=1)
                                    * torch.log(F.softmax(logits_per_text, dim=1)),
                                    dim=1
                                ).mean()
                                
                                # Get current weights for logging (average of both teachers)
                                current_weight = (id_teacher_enc.weight + ood_teacher_enc.weight) / 2
                                
                            else:
                                # Single teacher distillation (original logic)
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
                            
                                dist_loss_raw = -torch.sum(
                                    F.softmax(logits_per_image_t, dim=1)
                                    * torch.log(F.softmax(logits_per_image, dim=1))
                                    + F.softmax(logits_per_text_t, dim=1)
                                    * torch.log(F.softmax(logits_per_text, dim=1)),
                                    dim=1
                                ).mean()
                                
                                # Get current weight for logging
                                current_weight = teacher_enc.weight
                        
                        ft_clip_loss += args.distil_coef * dist_loss_raw
                        if isinstance(dist_loss_raw, torch.Tensor):
                            dist_loss_sum += args.distil_coef * dist_loss_raw.item()
            
            # Run gradient diagnostics before backward pass
            if grad_diagnostics is not None:
                # Create individual loss components dict for diagnostics
                loss_dict = create_loss_dict_for_diagnostics(
                    base_clip_loss_tensor, args,
                    oc_loss=oc_loss_raw,
                    crossf_loss=crossf_loss_raw,
                    sd_loss=dist_loss_raw if isinstance(dist_loss_raw, torch.Tensor) else None
                )
                
                # Get current performance metrics if available from previous epoch
                current_performance = {}
                if len(stats) > 0:
                    last_stats = stats[-1]
                    current_performance = {
                        'id_acc': last_stats.get('ImageNet Accuracy', 0),
                        'ood_acc': last_stats.get('Avg OOD Acc', 0),
                        'total_loss': last_stats.get('Avg Total Loss', 0),
                        'clip_loss': last_stats.get('Avg CLIP Loss', 0)
                    }
                
                # Log diagnostics
                grad_diagnostics.log_diagnostics(step, loss_dict, current_performance)

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
                if args.dual_teacher:
                    # Update both teachers with their respective frequencies
                    
                    # ID Teacher (EMA) update using ema_up_freq
                    teacher_update_stats['id_updated'] = False
                    if args.use_old_ema:
                        id_teacher_enc.update(step)
                        teacher_update_stats['id_updated'] = True
                    else:
                        if args.ema_up_freq <= 0:
                            id_teacher_enc.update()
                            teacher_update_stats['id_updated'] = True
                        else:
                            if ((step % args.ema_up_freq) == 0) or (step == num_batches * args.epochs - 1):
                                id_teacher_enc.update()
                                teacher_update_stats['id_updated'] = True
                    
                    # OOD Teacher (BMA) update using bma_up_freq
                    teacher_update_stats['ood_updated'] = False
                    if args.bma_up_freq <= 0:
                        # Update BMA teacher every step (default behavior)
                        ood_teacher_enc.update()
                        teacher_update_stats['ood_updated'] = True
                    else:
                        # Update BMA teacher at specified frequency
                        if ((step % args.bma_up_freq) == 0) or (step == num_batches * args.epochs - 1):
                            ood_teacher_enc.update()
                            teacher_update_stats['ood_updated'] = True
                            
                elif args.use_old_ema:
                    # Single teacher - old implementation needs the global step parameter
                    teacher_enc.update(step)
                else:
                    # Single teacher - new implementation - handle update frequency here
                    if args.bma_up_freq <= 0:
                        # Update teacher every step
                        teacher_enc.update()
                    else:
                        # Update teacher at specified frequency
                        if ((step % args.bma_up_freq) == 0) or (step == num_batches * args.epochs - 1):
                            teacher_enc.update()

            # Track base CLIP loss
            base_clip_loss = ft_clip_loss.item()
            if args.cross_fnorm:
                base_clip_loss -= args.cross_fnorm * fnorm_val
            if args.l_orth_wv:
                base_clip_loss -= args.l_orth_wv * orth_val
            if args.distil_coef and isinstance(dist_loss_raw, torch.Tensor):
                base_clip_loss -= args.distil_coef * dist_loss_raw.item()
            clip_loss_sum += base_clip_loss

            id_carot_loss_sum += ft_clip_loss.item()
            
            # Accumulate dual-teacher statistics for epoch summary
            if args.dual_teacher and args.distil_coef:
                epoch_dual_teacher_stats['total_steps'] += 1
                if teacher_update_stats['id_updated']:
                    epoch_dual_teacher_stats['id_teacher_updates'] += 1
                if teacher_update_stats['ood_updated']:
                    epoch_dual_teacher_stats['ood_teacher_updates'] += 1
                    
                if step > 0:
                    epoch_dual_teacher_stats['total_id_selections'] += teacher_selection_stats['id_selected']
                    epoch_dual_teacher_stats['total_ood_selections'] += teacher_selection_stats['ood_selected']
                    epoch_dual_teacher_stats['total_agreements'] += teacher_selection_stats['agreement_override']
                    epoch_dual_teacher_stats['sum_entropy_id'] += entropy_stats['mean_entropy_id']
                    epoch_dual_teacher_stats['sum_entropy_ood'] += entropy_stats['mean_entropy_ood']
                    epoch_dual_teacher_stats['sum_entropy_diff'] += entropy_stats['entropy_diff']
                    epoch_dual_teacher_stats['sum_id_confidence'] += teacher_selection_stats['id_confidence_rate']
                    epoch_dual_teacher_stats['sum_ood_confidence'] += teacher_selection_stats['ood_confidence_rate']
                    epoch_dual_teacher_stats['sum_both_confident'] += teacher_selection_stats['both_confident_rate']
                    epoch_dual_teacher_stats['sum_agreement_rate'] += teacher_selection_stats['prediction_agreement_rate']
                    epoch_dual_teacher_stats['batch_count'] += 1

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
                
                # Add distillation loss if applicable
                if args.distil_coef and isinstance(dist_loss_raw, torch.Tensor):
                    # Calculate training progress for display
                    total_steps = num_batches * args.epochs
                    progress = step / total_steps
                    log_msg += f"\n\tDistillation Loss: {args.distil_coef * dist_loss_raw.item():.4f} (Raw: {dist_loss_raw.item():.4f})"
                    
                    if args.dual_teacher:
                        # Dual-teacher specific logging with enhanced metrics
                        total_selections = teacher_selection_stats['id_selected'] + teacher_selection_stats['ood_selected']
                        id_pct = (teacher_selection_stats['id_selected'] / total_selections * 100) if total_selections > 0 else 0
                        ood_pct = (teacher_selection_stats['ood_selected'] / total_selections * 100) if total_selections > 0 else 0
                        
                        # Enhanced console logging
                        log_msg += f"\n\tDual-Teacher Selection: ID={id_pct:.1f}%, OOD={ood_pct:.1f}%, Agreements={teacher_selection_stats['agreement_override']}"
                        
                        # Teacher update status
                        update_status = []
                        if teacher_update_stats['id_updated']:
                            update_status.append("ID✓")
                        if teacher_update_stats['ood_updated']:
                            update_status.append("OOD✓")
                        update_str = f"Updated: {', '.join(update_status) if update_status else 'None'}"
                        log_msg += f"\n\t{update_str} (EMA freq: {args.ema_up_freq}, BMA freq: {args.bma_up_freq})"
                        
                        # Only show entropy and detailed stats if we have meaningful data (step > 0 and dual-teacher active)
                        if step > 0 and entropy_stats['mean_entropy_id'] > 0:
                            # Teacher correctness information
                            log_msg += f"\n\tTeacher Accuracy - ID: {teacher_selection_stats['id_correct_rate']:.1%}, OOD: {teacher_selection_stats['ood_correct_rate']:.1%}"
                            log_msg += f"\n\tCorrectness Pattern - Both Correct: {teacher_selection_stats['both_correct_rate']:.1%}, Both Wrong: {teacher_selection_stats['both_wrong_rate']:.1%}, Disagree: {teacher_selection_stats['correctness_disagree_rate']:.1%}"
                            
                            # Selection breakdown
                            correctness_selections = teacher_selection_stats['correctness_override_count']
                            confidence_selections = teacher_selection_stats['id_selected_by_confidence'] + teacher_selection_stats['ood_selected_by_confidence']
                            total_active_selections = correctness_selections + confidence_selections
                            
                            if total_active_selections > 0:
                                correctness_pct = (correctness_selections / total_active_selections) * 100
                                confidence_pct = (confidence_selections / total_active_selections) * 100
                                log_msg += f"\n\tSelection Mode - Correctness: {correctness_pct:.1f}% ({correctness_selections}), Confidence: {confidence_pct:.1f}% ({confidence_selections})"
                                log_msg += f"\n\tCorrectness Wins - ID: {teacher_selection_stats['id_selected_by_correctness']}, OOD: {teacher_selection_stats['ood_selected_by_correctness']}"
                                log_msg += f"\n\tConfidence Wins - ID: {teacher_selection_stats['id_selected_by_confidence']}, OOD: {teacher_selection_stats['ood_selected_by_confidence']}"
                            
                            log_msg += f"\n\tEntropy - ID: {entropy_stats['mean_entropy_id']:.3f}, OOD: {entropy_stats['mean_entropy_ood']:.3f}, Diff: {entropy_stats['entropy_diff']:.3f}"
                            log_msg += f"\n\tConfidence Rates - ID: {teacher_selection_stats['id_confidence_rate']:.1%}, OOD: {teacher_selection_stats['ood_confidence_rate']:.1%}, Both: {teacher_selection_stats['both_confident_rate']:.1%}"
                            log_msg += f"\n\tPrediction Agreement: {teacher_selection_stats['prediction_agreement_rate']:.1%}, Avg Teacher Weight: {current_weight:.4f}"
                            
                            # Effectiveness summary
                            better_teacher = 'ID' if teacher_selection_stats['id_correct_rate'] > teacher_selection_stats['ood_correct_rate'] else 'OOD'
                            more_confident_teacher = 'ID' if entropy_stats['mean_entropy_id'] < entropy_stats['mean_entropy_ood'] else 'OOD'
                            log_msg += f"\n\tTeacher Summary: {better_teacher} more accurate, {more_confident_teacher} more confident"
                        else:
                            log_msg += f"\n\tAvg Teacher Weight: {current_weight:.4f} (dual-teacher warming up)"
                        
                        # Comprehensive wandb logging
                        wandb_update_dict = {
                            # Basic distillation metrics
                            "Distillation Loss": args.distil_coef * dist_loss_raw.item(),
                            "Distillation Raw Loss": dist_loss_raw.item(),
                            "Training Progress": progress,
                            
                            # Teacher selection metrics
                            "DualTeacher/ID_Selected_Pct": id_pct,
                            "DualTeacher/OOD_Selected_Pct": ood_pct,
                            "DualTeacher/Teacher_Agreements": teacher_selection_stats['agreement_override'],
                            "DualTeacher/Total_Samples": teacher_selection_stats['total_samples'],
                            "DualTeacher/Entropy_Threshold": args.entropy_threshold,
                            "DualTeacher/Step": step,
                            
                            # Teacher update frequencies and status
                            "DualTeacher/EMA_Update_Freq": args.ema_up_freq,
                            "DualTeacher/BMA_Update_Freq": args.bma_up_freq,
                            "DualTeacher/ID_Teacher_Updated": teacher_update_stats['id_updated'],
                            "DualTeacher/OOD_Teacher_Updated": teacher_update_stats['ood_updated'],
                            
                            # Teacher weights
                            "DualTeacher/Avg_Teacher_Weight": current_weight,
                            "DualTeacher/ID_Teacher_Weight": id_teacher_enc.weight if hasattr(id_teacher_enc, 'weight') else 0,
                            "DualTeacher/OOD_Teacher_Weight": ood_teacher_enc.weight if hasattr(ood_teacher_enc, 'weight') else 0,
                            "DualTeacher/Selection_Efficiency": max(id_pct, ood_pct),  # How dominant is the better teacher
                        }
                        
                        # Only add detailed entropy metrics if we have meaningful data
                        if step > 0 and entropy_stats['mean_entropy_id'] > 0:
                            wandb_update_dict.update({
                                # Entropy metrics
                                "DualTeacher/Mean_Entropy_ID": entropy_stats['mean_entropy_id'],
                                "DualTeacher/Mean_Entropy_OOD": entropy_stats['mean_entropy_ood'],
                                "DualTeacher/Entropy_Difference": entropy_stats['entropy_diff'],
                                
                                # Confidence and agreement metrics
                                "DualTeacher/ID_Confidence_Rate": teacher_selection_stats['id_confidence_rate'],
                                "DualTeacher/OOD_Confidence_Rate": teacher_selection_stats['ood_confidence_rate'],
                                "DualTeacher/Both_Confident_Rate": teacher_selection_stats['both_confident_rate'],
                                "DualTeacher/Prediction_Agreement_Rate": teacher_selection_stats['prediction_agreement_rate'],
                                
                                # NEW: Correctness-based metrics
                                "DualTeacher/ID_Correct_Rate": teacher_selection_stats['id_correct_rate'],
                                "DualTeacher/OOD_Correct_Rate": teacher_selection_stats['ood_correct_rate'],
                                "DualTeacher/Both_Correct_Rate": teacher_selection_stats['both_correct_rate'],
                                "DualTeacher/Both_Wrong_Rate": teacher_selection_stats['both_wrong_rate'],
                                "DualTeacher/Correctness_Disagree_Rate": teacher_selection_stats['correctness_disagree_rate'],
                                
                                # Selection breakdown metrics
                                "DualTeacher/Correctness_Override_Count": teacher_selection_stats['correctness_override_count'],
                                "DualTeacher/ID_Selected_By_Correctness": teacher_selection_stats['id_selected_by_correctness'],
                                "DualTeacher/OOD_Selected_By_Correctness": teacher_selection_stats['ood_selected_by_correctness'],
                                "DualTeacher/ID_Selected_By_Confidence": teacher_selection_stats['id_selected_by_confidence'],
                                "DualTeacher/OOD_Selected_By_Confidence": teacher_selection_stats['ood_selected_by_confidence'],
                                
                                # Derived metrics for analysis
                                "DualTeacher/Correctness_Selection_Pct": (teacher_selection_stats['correctness_override_count'] / max(teacher_selection_stats['total_samples'], 1)) * 100,
                                "DualTeacher/ID_Better_Accuracy": 1 if teacher_selection_stats['id_correct_rate'] > teacher_selection_stats['ood_correct_rate'] else 0,
                                "DualTeacher/Accuracy_Difference": teacher_selection_stats['id_correct_rate'] - teacher_selection_stats['ood_correct_rate'],
                                
                                # Additional analysis metrics
                                "DualTeacher/Entropy_Ratio_ID_to_OOD": entropy_stats['mean_entropy_id'] / max(entropy_stats['mean_entropy_ood'], 1e-8),
                                "DualTeacher/Teacher_Dominance": 1 if entropy_stats['mean_entropy_id'] < entropy_stats['mean_entropy_ood'] else -1,
                            })
                        
                        wandb_log.update(wandb_update_dict)
                    elif args.use_old_ema:
                        log_msg += f"\n\tEMA Momentum: {current_weight:.4f} (progress: {progress:.2%})"
                        wandb_log.update({
                            "Distillation Loss": args.distil_coef * dist_loss_raw.item(),
                            "Distillation Raw Loss": dist_loss_raw.item(),
                            "EMA Momentum": current_weight,
                            "Training Progress": progress,
                        })
                    else:
                        log_msg += f"\n\tBeta Momentum: {current_weight:.4f} (progress: {progress:.2%})"
                        wandb_log.update({
                            "Distillation Loss": args.distil_coef * dist_loss_raw.item(),
                            "Distillation Raw Loss": dist_loss_raw.item(),
                            "Beta Momentum": current_weight,
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

        # Compute averages at the end of each epoch
        id_carot_loss_avg = id_carot_loss_sum / num_batches
        clip_loss_avg = clip_loss_sum / num_batches

        # Update epoch stats with all metrics
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

        if args.distil_coef:
            dist_loss_avg = dist_loss_sum / num_batches
            epoch_stats["Avg Distillation Loss"] = round(dist_loss_avg, 4)
            logger.info(f"  Avg Distillation Loss: {dist_loss_avg:.4f}")
            
            if args.dual_teacher:
                logger.info(f"  Dual-Teacher Mode: ID Teacher (EMA) + OOD Teacher (BMA)")
                logger.info(f"  Entropy Threshold: {args.entropy_threshold}")
                
                # Calculate and log epoch averages for dual-teacher metrics
                if epoch_dual_teacher_stats['total_steps'] > 0:
                    # Teacher update statistics
                    id_update_rate = epoch_dual_teacher_stats['id_teacher_updates'] / epoch_dual_teacher_stats['total_steps']
                    ood_update_rate = epoch_dual_teacher_stats['ood_teacher_updates'] / epoch_dual_teacher_stats['total_steps']
                    
                    logger.info(f"  Teacher Updates: ID={epoch_dual_teacher_stats['id_teacher_updates']}/{epoch_dual_teacher_stats['total_steps']} ({id_update_rate:.1%}), OOD={epoch_dual_teacher_stats['ood_teacher_updates']}/{epoch_dual_teacher_stats['total_steps']} ({ood_update_rate:.1%})")
                    
                    if epoch_dual_teacher_stats['batch_count'] > 0:
                        total_selections = epoch_dual_teacher_stats['total_id_selections'] + epoch_dual_teacher_stats['total_ood_selections']
                        epoch_id_pct = (epoch_dual_teacher_stats['total_id_selections'] / total_selections * 100) if total_selections > 0 else 0
                        epoch_ood_pct = (epoch_dual_teacher_stats['total_ood_selections'] / total_selections * 100) if total_selections > 0 else 0
                        
                        avg_entropy_id = epoch_dual_teacher_stats['sum_entropy_id'] / epoch_dual_teacher_stats['batch_count']
                        avg_entropy_ood = epoch_dual_teacher_stats['sum_entropy_ood'] / epoch_dual_teacher_stats['batch_count']
                        avg_entropy_diff = epoch_dual_teacher_stats['sum_entropy_diff'] / epoch_dual_teacher_stats['batch_count']
                        avg_id_confidence = epoch_dual_teacher_stats['sum_id_confidence'] / epoch_dual_teacher_stats['batch_count']
                        avg_ood_confidence = epoch_dual_teacher_stats['sum_ood_confidence'] / epoch_dual_teacher_stats['batch_count']
                        avg_both_confident = epoch_dual_teacher_stats['sum_both_confident'] / epoch_dual_teacher_stats['batch_count']
                        avg_agreement_rate = epoch_dual_teacher_stats['sum_agreement_rate'] / epoch_dual_teacher_stats['batch_count']
                        
                        logger.info(f"  Epoch Teacher Selection: ID={epoch_id_pct:.1f}%, OOD={epoch_ood_pct:.1f}%")
                        logger.info(f"  Epoch Avg Entropy: ID={avg_entropy_id:.3f}, OOD={avg_entropy_ood:.3f}, Diff={avg_entropy_diff:.3f}")
                        logger.info(f"  Epoch Confidence Rates: ID={avg_id_confidence:.1%}, OOD={avg_ood_confidence:.1%}, Both={avg_both_confident:.1%}")
                        logger.info(f"  Epoch Agreement Rate: {avg_agreement_rate:.1%}, Total Agreements: {epoch_dual_teacher_stats['total_agreements']}")
                        
                        # Add detailed epoch statistics to epoch_stats for wandb
                        epoch_stats.update({
                            "DualTeacher_Epoch/ID_Selected_Pct": round(epoch_id_pct, 2),
                            "DualTeacher_Epoch/OOD_Selected_Pct": round(epoch_ood_pct, 2),
                            "DualTeacher_Epoch/Avg_Entropy_ID": round(avg_entropy_id, 4),
                            "DualTeacher_Epoch/Avg_Entropy_OOD": round(avg_entropy_ood, 4),
                            "DualTeacher_Epoch/Avg_Entropy_Diff": round(avg_entropy_diff, 4),
                            "DualTeacher_Epoch/Avg_ID_Confidence_Rate": round(avg_id_confidence, 4),
                            "DualTeacher_Epoch/Avg_OOD_Confidence_Rate": round(avg_ood_confidence, 4),
                            "DualTeacher_Epoch/Avg_Both_Confident_Rate": round(avg_both_confident, 4),
                            "DualTeacher_Epoch/Avg_Agreement_Rate": round(avg_agreement_rate, 4),
                            "DualTeacher_Epoch/Total_ID_Selections": epoch_dual_teacher_stats['total_id_selections'],
                            "DualTeacher_Epoch/Total_OOD_Selections": epoch_dual_teacher_stats['total_ood_selections'],
                            "DualTeacher_Epoch/Total_Agreements": epoch_dual_teacher_stats['total_agreements'],
                            "DualTeacher_Epoch/Processed_Batches": epoch_dual_teacher_stats['batch_count'],
                        })
                    
                    # Add teacher update statistics to wandb
                    epoch_stats.update({
                        "DualTeacher_Epoch/ID_Teacher_Updates": epoch_dual_teacher_stats['id_teacher_updates'],
                        "DualTeacher_Epoch/OOD_Teacher_Updates": epoch_dual_teacher_stats['ood_teacher_updates'],
                        "DualTeacher_Epoch/ID_Update_Rate": round(id_update_rate, 4),
                        "DualTeacher_Epoch/OOD_Update_Rate": round(ood_update_rate, 4),
                        "DualTeacher_Epoch/Total_Steps": epoch_dual_teacher_stats['total_steps'],
                        "DualTeacher_Epoch/EMA_Update_Freq_Setting": args.ema_up_freq,
                        "DualTeacher_Epoch/BMA_Update_Freq_Setting": args.bma_up_freq,
                    })

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

            #! save the teacher(s)
            if args.distil_coef:
                if args.dual_teacher:
                    # Save both teachers
                    id_teacher_path = os.path.join(args.save, f"checkpoint_{epoch+1}_ID_teacher.pt")
                    ood_teacher_path = os.path.join(args.save, f"checkpoint_{epoch+1}_OOD_teacher.pt")
                    logger.info("Saving ID teacher to " + str(id_teacher_path))
                    logger.info("Saving OOD teacher to " + str(ood_teacher_path))
                    try:
                        id_teacher_enc.save(id_teacher_path)
                        ood_teacher_enc.save(ood_teacher_path)
                    except Exception as e:
                        print("============================")
                        print(f"Error occurred during dual-teacher model saving: {e}")
                        print("============================")
                else:
                    # Save single teacher
                    ema_model_path = os.path.join(args.save, f"checkpoint_{epoch+1}_EMA.pt")
                    logger.info("Saving teacher to " + str(ema_model_path))
                    try:
                        teacher_enc.save(ema_model_path)
                    except Exception as e:
                        print("============================")
                        print(f"Error occurred during teacher model saving: {e}")
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
            + f"_run{args.run}"
        )
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + "/stats.tsv", sep="\t")

        #! wandb logging
        wandb.log({k: v for k, v in epoch_stats.items()})

    # Final gradient diagnostics summary
    if grad_diagnostics is not None:
        logger.info("Generating final gradient diagnostics summary")
        conflicts = grad_diagnostics.get_gradient_conflicts_summary()
        if conflicts:
            logger.info("Summary of gradient conflicts during training:")
            for conflict_name, conflict_data in conflicts.items():
                logger.info(f"  {conflict_name}: {len(conflict_data)} instances")
                if conflict_data:
                    min_val = min(conflict_data, key=lambda x: x[1])
                    max_val = max(conflict_data, key=lambda x: x[1])
                    logger.info(f"    Range: {min_val[1]:.3f} to {max_val[1]:.3f}")
                    logger.info(f"    First occurrence: step {conflict_data[0][0]}")
        else:
            logger.info("No significant gradient conflicts detected during training")

    if args.save is not None:
        return model_path
