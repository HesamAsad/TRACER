import os
import time
from tqdm.auto import tqdm
import wandb

import torch
# torch.set_default_dtype(torch.bfloat16)  # Removed: causes RN50 BatchNorm dtype issues with mixed precision
from torch.nn import functional as F
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.models.eval import evaluate
from src.models.utils import cosine_lr, cosine_grad_norm_scheduler, apply_layer_freezing, fix_batchnorm_dtype_for_mixed_precision
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.laion import get_data
from src.models.beta_moving_average import (
    GeneralMovingAverage,
    create_beta_weight_function,
    ExponentialMovingAverage,
    create_linear_warmup_ema_momentum,
)
from src.models.clip_knowledge_distillation import create_clip_kd_module, calculate_teacher_statistics
from src.models.geodesic_mix import sph_inter, sample_beta_lambda
from src.datasets_.spatial_captions import SpatialCaptionIndex, log_spatial_index_stats
import src.datasets_ as datasets


def tracer_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."

    logger.info("Fine-tuning Using Tracer Loss")
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

    spatial_index = None
    if getattr(args, "use_spatial_captions", 0) and getattr(args, "spatial_captions_jsonl", None):
        spatial_index = SpatialCaptionIndex(args.spatial_captions_jsonl, args.data_location)
        msg = f"Loaded {len(spatial_index)} spatial captions from {args.spatial_captions_jsonl}"
        print(f"[spatial-captions] {msg}", flush=True)
        logger.info(msg)

    img_text_data = get_data(
        args,
        (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0,
        spatial_caption_index=spatial_index,
    )
    assert len(img_text_data), "At least one train or eval dataset must be specified."
    ft_dataloader = img_text_data["train_ft"].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    use_spatial = bool(getattr(args, "use_spatial_captions", 0)) and (spatial_index is not None)
    if getattr(args, "use_spatial_captions", 0) and spatial_index is None:
        logger.warning(
            "--use-spatial-captions is set but --spatial-captions-jsonl was not provided; "
            "falling back to template-only training."
        )

    if spatial_index is not None:
        try:
            log_spatial_index_stats(spatial_index, ft_dataloader.dataset.images, logger=logger)
        except Exception as e:
            logger.warning(f"Could not compute spatial-caption match rate: {e}")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler('cuda')

    if args.clip_load is not None:
        model = model.load(args.clip_load)

    if args.distil_coef:
        # Create teacher encoder: EMA (if enabled) or Beta-MA (default)
        total_iterations = args.epochs * num_batches
        teacher_model = model.cuda()
        if args.use_fp16:
            teacher_model = fix_batchnorm_dtype_for_mixed_precision(teacher_model)

        if getattr(args, 'ema_teacher', False):
            # EMA with linear warmup momentum schedule
            get_momentum_fn = create_linear_warmup_ema_momentum(
                src_momentum=args.m_sche_src,
                tar_momentum=args.m_sche_tar,
                warmup_ratio=args.m_warm_up,
                total_iterations=total_iterations,
            )
            # Default EMA update frequency to 500 if not provided (>0 keeps user's value)
            ema_up_freq = args.ema_up_freq if args.ema_up_freq > 0 else 500
            teacher_enc = ExponentialMovingAverage(
                teacher_model, get_momentum_fn, update_frequency=ema_up_freq
            )
        else:
            weight_func = create_beta_weight_function(args.beta, total_iterations)
            teacher_enc = GeneralMovingAverage(teacher_model, weight_func)

        # Effective update frequency used later for gating teacher updates
        ema_up_freq = args.ema_up_freq if not getattr(args, 'ema_teacher', False) else (args.ema_up_freq if args.ema_up_freq > 0 else 500)
    
    model = model.cuda()
    
    # Fix BatchNorm dtype for mixed precision training compatibility
    if args.use_fp16:
        model = fix_batchnorm_dtype_for_mixed_precision(model)
        logger.info("Fixed BatchNorm buffer dtypes for mixed precision training")

    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info("Using devices" + str(devices))

    model = torch.nn.DataParallel(model, device_ids=devices)

    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)
    classification_head.train()
    model.train()

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
    
    # Initialize comprehensive knowledge distillation module
    kd_module = None
    if any([args.alpha_crd, args.alpha_fd, args.alpha_mfd, args.alpha_gd, 
            args.alpha_icl, args.alpha_afd, args.alpha_cross_kd, args.alpha_temp_distil]):
        # Get embedding dimensions from model
        if hasattr(model.module.model, 'visual') and hasattr(model.module.model.visual, 'output_dim'):
            embed_dim = model.module.model.visual.output_dim
        elif hasattr(model.module.model, 'embed_dim'):
            embed_dim = model.module.model.embed_dim
        else:
            embed_dim = 512  # Default for most CLIP models
        
        kd_module = create_clip_kd_module(args, embed_dim=embed_dim)
        kd_module = kd_module.cuda()
        
        # Add KD module parameters to optimizer if it has trainable parameters
        if hasattr(kd_module, 'visual_fusion_proj') or hasattr(kd_module, 'img_align_proj'):
            kd_params = list(kd_module.parameters())
            if kd_params:
                params.extend([p for p in kd_params if p.requires_grad])
                logger.info(f"Added {len([p for p in kd_params if p.requires_grad])} KD module parameters to optimizer")
    
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

    for epoch in tqdm(range(0, args.epochs), desc="Epochs"):
        print("\nEpoch : ", epoch)
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        # Initialize tracking variables for epoch statistics
        id_tracer_loss_sum = 0
        fnorm_loss_sum = 0
        orth_loss_sum = 0
        dist_loss_sum = 0
        clip_loss_sum = 0
        supcon_logged_this_epoch = False  # Track if we've logged supervised contrastive info this epoch
        
        # Initialize KD loss tracking
        kd_loss_sums = {
            'crd_loss': 0.0,
            'fd_loss': 0.0,
            'mfd_loss': 0.0,
            'gd_loss': 0.0,
            'icl_loss': 0.0,
            'afd_loss': 0.0,
            'cross_kd_loss': 0.0,
            'temp_distil_loss': 0.0,
            'total_kd_loss': 0.0
        }

        m2_loss_sum = 0.0
        spatial_match_sum = 0.0
        spatial_match_count = 0
        sanity_done = False
        
        # Initialize teacher statistics tracking
        teacher_stats_accumulator = {}
        teacher_stats_count = 0
        
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
            
            ft_labels = None
            use_supcon = False
            ft_text_tmpl = None
            ft_text_sp = None
            has_spatial = None

            if use_spatial:
                if len(ft_batch) == 5:
                    ft_image, ft_text_tmpl, ft_text_sp, has_spatial, ft_labels = ft_batch
                    ft_labels = ft_labels.cuda()
                    use_supcon = True
                    if not supcon_logged_this_epoch:
                        logger.info(f"Using supervised CLIP loss with labels for epoch {epoch}")
                        supcon_logged_this_epoch = True
                else:
                    ft_image, ft_text_tmpl, ft_text_sp, has_spatial = ft_batch
                ft_image = ft_image.cuda()
                ft_text_tmpl = ft_text_tmpl.cuda()
                ft_text_sp = ft_text_sp.cuda()
                has_spatial = has_spatial.cuda()
                # Downstream blocks (teacher forward etc.) still reference ft_text.
                ft_text = ft_text_tmpl
            else:
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

            lam_tt_eff = None
            txt_feat_tmpl = None
            txt_feat_sp = None
            txt_feat_mixed = None
            m2_loss_value = 0.0

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32):
                if use_spatial:
                    # Encode image once; call encode_text directly for the spatial pass
                    # so we don't re-run the (much larger) image encoder.
                    img_feat, txt_feat_tmpl, logit_scale2 = model(ft_image, ft_text_tmpl)
                    core_model = model.module.model if hasattr(model, "module") else model.model
                    try:
                        txt_feat_sp = core_model.encode_text(ft_text_sp, normalize=True)
                    except TypeError:
                        # OpenAI CLIP's encode_text has no `normalize` kwarg.
                        txt_feat_sp = core_model.encode_text(ft_text_sp)
                        txt_feat_sp = txt_feat_sp / (txt_feat_sp.norm(dim=-1, keepdim=True) + 1e-12)

                    B = ft_image.size(0)
                    lam_tt = sample_beta_lambda(
                        args.alpha_tt_mix, args.beta_tt_mix,
                        ft_image.device, bool(args.tt_per_sample), B=B,
                    )
                    # lambda=1 pins samples without a spatial caption to pure template.
                    has_sp_bool = has_spatial.bool()
                    if lam_tt.numel() == B:
                        lam_tt_eff = torch.where(has_sp_bool, lam_tt, torch.ones_like(lam_tt))
                    else:
                        lam_tt_eff = lam_tt if has_sp_bool.all() else torch.ones_like(lam_tt)

                    txt_feat_mixed = sph_inter(txt_feat_tmpl, txt_feat_sp, lam_tt_eff)

                    if lam_tt.numel() == B:
                        mask = has_sp_bool.view(-1, 1).expand_as(txt_feat_mixed)
                        txt_feat_mixed = torch.where(mask, txt_feat_mixed, txt_feat_tmpl)

                    ft_image_features = img_feat
                    ft_text_features = txt_feat_mixed
                    spatial_match_sum += has_sp_bool.float().mean().item()
                    spatial_match_count += 1
                else:
                    ft_image_features, ft_text_features, logit_scale2 = model(
                        ft_image, ft_text
                    )

                lscale = logit_scale2 if len(devices) == 1 else logit_scale2[0]

                ft_clip_loss, logits_per_image, logits_per_text = clip_loss_fn(
                    ft_image_features, ft_text_features, lscale
                )

                if use_spatial and args.beta_mix_coef > 0:
                    if args.beta_mix_target == "spatial":
                        target_text = txt_feat_sp
                    elif args.beta_mix_target == "template":
                        target_text = txt_feat_tmpl
                    elif args.beta_mix_target == "alpha_mixed":
                        target_text = txt_feat_mixed
                    else:
                        raise ValueError(f"unknown --beta-mix-target: {args.beta_mix_target}")

                    B = ft_image_features.size(0)
                    lam_it = sample_beta_lambda(
                        args.alpha_it_mix, args.alpha_it_mix,
                        ft_image_features.device, bool(args.it_per_sample), B=B,
                    )
                    mixed_it = sph_inter(ft_image_features, target_text, lam_it)

                    # open_clip's `logit_scale2` is already .exp()'d, so `lscale` IS the
                    # multiplier (~100). tau2>0 uses the literal value as m_tau for the
                    # off-diagonal negatives (reference default 0.01).
                    scale_pos = lscale
                    scale_neg = (
                        torch.tensor(args.tau2, device=ft_image_features.device, dtype=ft_image_features.dtype)
                        if args.tau2 > 0 else lscale
                    )

                    I_mat = torch.eye(B, device=ft_image_features.device, dtype=ft_image_features.dtype)
                    off_diag = 1.0 - I_mat

                    logits_pos_i2t = scale_pos * (ft_image_features @ txt_feat_tmpl.T)
                    logits_pos_t2i = scale_pos * (txt_feat_tmpl @ ft_image_features.T)
                    logits_neg_i2t = scale_neg * (ft_image_features @ mixed_it.T)
                    logits_neg_t2i = scale_neg * (txt_feat_tmpl @ mixed_it.T)

                    logits_m2_i2t = logits_pos_i2t * I_mat + logits_neg_i2t * off_diag
                    logits_m2_t2i = logits_pos_t2i * I_mat + logits_neg_t2i * off_diag

                    labels = torch.arange(B, device=ft_image_features.device)
                    m2_loss = 0.5 * (
                        F.cross_entropy(logits_m2_i2t, labels)
                        + F.cross_entropy(logits_m2_t2i, labels)
                    )

                    ft_clip_loss = ft_clip_loss + args.beta_mix_coef * m2_loss
                    m2_loss_value = m2_loss.item()
                    m2_loss_sum += args.beta_mix_coef * m2_loss_value

                    if args.sanity_check and not sanity_done:
                        with torch.no_grad():
                            hard_sim = (ft_image_features * mixed_it).sum(-1).mean().item()
                            perm = torch.randperm(B, device=ft_image_features.device)
                            while B > 1 and (perm == torch.arange(B, device=perm.device)).any():
                                perm = torch.randperm(B, device=ft_image_features.device)
                            rand_sim = (ft_image_features * txt_feat_tmpl[perm]).sum(-1).mean().item()
                            msg = (
                                f"[sanity] scenario-beta hardness: mix_sim={hard_sim:.4f} "
                                f"vs random-offdiag_sim={rand_sim:.4f} "
                                f"(mix should be >= random for hardness to help)"
                            )
                            print(msg, flush=True)
                            logger.info(msg)

                if args.sanity_check and use_spatial and not sanity_done:
                    with torch.no_grad():
                        norms = txt_feat_mixed.norm(dim=-1)
                        assert torch.all((norms - 1.0).abs() < 1e-2), (
                            f"[sanity] mixed text not unit norm: min={norms.min()} max={norms.max()}"
                        )
                        test_m1 = sph_inter(txt_feat_tmpl, txt_feat_sp, 1.0)
                        test_m0 = sph_inter(txt_feat_tmpl, txt_feat_sp, 0.0)
                        err1 = (test_m1 - txt_feat_tmpl).norm(dim=-1).max().item()
                        err0 = (test_m0 - txt_feat_sp).norm(dim=-1).max().item()
                        msg = (
                            f"[sanity] endpoint recovery: |mix(s=1)-tmpl|={err1:.2e}, "
                            f"|mix(s=0)-sp|={err0:.2e}"
                        )
                        print(msg, flush=True)
                        logger.info(msg)
                        assert err1 < 1e-2 and err0 < 1e-2, "[sanity] endpoint recovery failed"
                    sanity_done = True

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
                        covv = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.visual.attnpool.c_proj.weight
                    else:
                        covv = model.module.model.visual.proj.T @ model.module.model.visual.proj
                    orth_val = ((covv - torch.eye(covv.shape[0], device=covv.device))**2).sum()**(1/2)
                    ft_clip_loss += args.l_orth_wv * orth_val
                    orth_val = orth_val.item()
                    orth_loss_sum += args.l_orth_wv * orth_val

            #! self-distillation flag
            dist_loss, current_weight = torch.tensor(0), 0.0
            teacher_stats = {}  # Dictionary to store teacher statistics
            
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
                            
                            # Calculate comprehensive teacher statistics first (using original logits)
                            teacher_stats = calculate_teacher_statistics(
                                logits_per_image_t, logits_per_text_t,
                                logits_per_image, logits_per_text
                            )
                            
                            # Apply temperature scaling to teacher logits for better distillation
                            distillation_temperature = getattr(args, 'distillation_temperature', 1.0)

                            # Add temperature scaling info to teacher stats
                            teacher_stats['distillation_temperature'] = distillation_temperature

                            if distillation_temperature != 1.0:
                                logits_per_image_t_scaled = logits_per_image_t / distillation_temperature
                                logits_per_text_t_scaled = logits_per_text_t / distillation_temperature
                                
                                # Calculate post-temperature scaling statistics
                                post_temp_stats = calculate_teacher_statistics(
                                    logits_per_image_t_scaled, logits_per_text_t_scaled,
                                    logits_per_image, logits_per_text
                                )
                                
                                # Add post-temperature stats with prefix
                                for key, value in post_temp_stats.items():
                                    if key.startswith('teacher_'):
                                        teacher_stats[f'post_temp_{key}'] = value
                            else:
                                logits_per_image_t_scaled = logits_per_image_t
                                logits_per_text_t_scaled = logits_per_text_t
                        
                        # Use temperature-scaled logits for distillation loss
                        # Only apply basic distillation if kd_module is not being used
                        if kd_module is None:
                            dist_loss = -torch.sum(
                                F.softmax(logits_per_image_t_scaled, dim=1)
                                * F.log_softmax(logits_per_image, dim=1)
                                + F.softmax(logits_per_text_t_scaled, dim=1)
                                * F.log_softmax(logits_per_text, dim=1),
                                dim=1
                            ).mean()
                            
                            ft_clip_loss += args.distil_coef * dist_loss
                            if isinstance(dist_loss, torch.Tensor):
                                dist_loss_sum += args.distil_coef * dist_loss.item()
                        
                        # Get current momentum/weight for logging
                        current_weight = getattr(teacher_enc, 'weight', getattr(teacher_enc, 'momentum', 0.0))
                        
                        # Apply comprehensive knowledge distillation methods
                        if kd_module is not None:
                            # Create labels for KD methods that need them
                            batch_size = ft_image_features.size(0)
                            kd_labels = torch.arange(batch_size, device=ft_image_features.device)
                            
                            # Get distillation temperature
                            kd_temperature = teacher_stats.get('distillation_temperature', distillation_temperature)
                            
                            # Compute all KD losses
                            kd_losses = kd_module(
                                student_img_features=ft_image_features,
                                student_txt_features=ft_text_features,
                                teacher_img_features=ft_image_features_t,
                                teacher_txt_features=ft_text_features_t,
                                student_logit_scale=lscale,
                                teacher_logit_scale=logit_scale_t,
                                labels=kd_labels,
                                temperature=kd_temperature
                            )
                            
                            # Add KD losses to total loss
                            if kd_losses['total_kd_loss'] > 0:
                                ft_clip_loss += args.distil_coef * kd_losses['total_kd_loss']
                                
                                # Update teacher statistics with KD loss info
                                for loss_name, loss_value in kd_losses.items():
                                    if isinstance(loss_value, torch.Tensor):
                                        teacher_stats[f'kd_{loss_name}'] = loss_value.item()
                                        # Track KD losses for epoch summary
                                        if loss_name in kd_loss_sums:
                                            kd_loss_sums[loss_name] += loss_value.item()
                                    else:
                                        teacher_stats[f'kd_{loss_name}'] = loss_value
                                        # Track KD losses for epoch summary
                                        if loss_name in kd_loss_sums:
                                            kd_loss_sums[loss_name] += loss_value
                        
                        # Accumulate teacher statistics for epoch summary
                        if teacher_stats:
                            teacher_stats_count += 1
                            for key, value in teacher_stats.items():
                                if key not in teacher_stats_accumulator:
                                    teacher_stats_accumulator[key] = 0
                                teacher_stats_accumulator[key] += value
                else:
                    # At step 0, teacher hasn't been initialized yet
                    logger.info("Teacher model statistics not available at step 0")

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
                    # Deprecated logic parity: do NOT update if freq <= 0, else update at multiples; use global_step for schedule
                    teacher_update_freq = args.ema_up_freq if args.ema_up_freq > 0 else 500
                    if teacher_update_freq > 0 and (((step % teacher_update_freq) == 0) or (step == total_steps)):
                        teacher_enc.update(global_step=step)
                else:
                    # BMA behavior: if freq <= 0, update every step; else multiples
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
            if args.distil_coef and kd_module is None and isinstance(dist_loss, torch.Tensor):
                base_clip_loss -= args.distil_coef * dist_loss.item()
            if use_spatial and args.beta_mix_coef > 0:
                base_clip_loss -= args.beta_mix_coef * m2_loss_value
            clip_loss_sum += base_clip_loss

            id_tracer_loss_sum += ft_clip_loss.item()

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
                
                # Add distillation loss if applicable (only for basic distillation)
                if args.distil_coef and kd_module is None and isinstance(dist_loss, torch.Tensor):
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
                elif args.distil_coef and kd_module is not None:
                    # For kd_module case, just log beta momentum and progress
                    total_steps = num_batches * args.epochs
                    progress = step / total_steps
                    momentum_label = "EMA Momentum" if getattr(args, 'ema_teacher', False) else "Beta Momentum"
                    log_msg += f"\n\t{momentum_label}: {current_weight:.4f} (progress: {progress:.2%})"
                    wandb_log.update({
                        (momentum_label): current_weight,
                        "Training Progress": progress,
                    })
                    
                    # Add teacher statistics if available
                    if teacher_stats:
                        # Add key teacher metrics to log message
                        log_msg += f"\n\tTeacher Entropy: {teacher_stats.get('teacher_avg_entropy_combined', 0):.4f}"
                        log_msg += f"\n\tTeacher Confidence: {teacher_stats.get('teacher_avg_confidence_combined', 0):.4f}"
                        log_msg += f"\n\tTeacher GT Prob: {teacher_stats.get('teacher_gt_prob_combined', 0):.4f}"
                        log_msg += f"\n\tTeacher-Student Agreement: {teacher_stats.get('teacher_student_agreement_combined', 0):.4f}"
                        log_msg += f"\n\tTeacher-Student KL: {teacher_stats.get('teacher_student_kl_combined', 0):.4f}"
                        
                        # Add temperature scaling info
                        if 'distillation_temperature' in teacher_stats:
                            log_msg += f"\n\tDistillation Temperature: {teacher_stats.get('distillation_temperature', 0):.2f}"
                            log_msg += f"\n\tEffective/Distill Temp Ratio: {teacher_stats.get('teacher_effective_temp_vs_distill_temp', 0):.2f}"
                        
                        # Add all teacher statistics to wandb
                        teacher_wandb_log = {}
                        for key, value in teacher_stats.items():
                            teacher_wandb_log[f"Teacher_{key}"] = value
                        wandb_log.update(teacher_wandb_log)
                        
                        # Log KD losses if available
                        kd_log_msg = ""
                        kd_losses_active = False
                        for key, value in teacher_stats.items():
                            if key.startswith('kd_') and key.endswith('_loss'):
                                kd_losses_active = True
                                loss_name = key[3:]  # Remove 'kd_' prefix
                                if loss_name == 'total_kd_loss':
                                    kd_log_msg += f"\n\tTotal KD Loss: {value:.4f}"
                                elif loss_name.endswith('_loss'):
                                    method_name = loss_name[:-5].upper()  # Remove '_loss' suffix and uppercase
                                    kd_log_msg += f"\n\t{method_name}: {value:.4f}"
                        
                        if kd_log_msg and kd_losses_active:
                            log_msg += kd_log_msg
                
                # Add learning rate
                current_lr = optimizer.param_groups[0]['lr']
                log_msg += f"\n\tLearning Rate: {current_lr:.6f}"
                wandb_log.update({"Learning Rate": current_lr})
                
                # Add logit scale
                log_msg += f"\n\tLogit Scale: {lscale.exp().item():.4f}"
                wandb_log.update({"Logit Scale": lscale.exp().item()})

                if use_spatial and lam_tt_eff is not None and txt_feat_mixed is not None:
                    match_rate = has_spatial.float().mean().item()
                    lam_mean = lam_tt_eff.float().mean().item()
                    lam_std = (
                        lam_tt_eff.float().std().item() if lam_tt_eff.numel() > 1 else 0.0
                    )
                    cos_tmpl = (txt_feat_mixed * txt_feat_tmpl).sum(-1).mean().item()
                    cos_sp = (txt_feat_mixed * txt_feat_sp).sum(-1).mean().item()
                    log_msg += (
                        f"\n\tSpatial match rate: {match_rate:.3f}"
                        f"\n\tAlpha lambda (mean/std): {lam_mean:.3f} / {lam_std:.3f}"
                        f"\n\tText mixed-vs-(tmpl/sp) cos: {cos_tmpl:.3f} / {cos_sp:.3f}"
                    )
                    wandb_log.update({
                        "Spatial match rate": match_rate,
                        "Alpha lambda mean": lam_mean,
                        "Alpha lambda std": lam_std,
                        "Text mixed-vs-template cos": cos_tmpl,
                        "Text mixed-vs-spatial cos": cos_sp,
                    })
                    if args.beta_mix_coef > 0:
                        log_msg += (
                            f"\n\tScenario-beta m2 loss: {m2_loss_value:.4f} "
                            f"(weighted: {args.beta_mix_coef * m2_loss_value:.4f})"
                        )
                        wandb_log.update({
                            "Scenario-beta m2 loss": m2_loss_value,
                            "Scenario-beta weighted": args.beta_mix_coef * m2_loss_value,
                        })

                logger.info(log_msg)
                wandb.log(wandb_log)

        # Compute averages at the end of each epoch
        id_tracer_loss_avg = id_tracer_loss_sum / num_batches
        clip_loss_avg = clip_loss_sum / num_batches

        # Update epoch stats with all metrics
        epoch_stats["Avg Total Loss"] = round(id_tracer_loss_avg, 4)
        epoch_stats["Avg CLIP Loss"] = round(clip_loss_avg, 4)

        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Avg Total Loss: {id_tracer_loss_avg:.4f}")
        logger.info(f"  Avg CLIP Loss: {clip_loss_avg:.4f}")

        if args.cross_fnorm:
            fnorm_loss_avg = fnorm_loss_sum / num_batches
            epoch_stats["Avg Cross F-norm Loss"] = round(fnorm_loss_avg, 4)
            logger.info(f"  Avg Cross F-norm Loss: {fnorm_loss_avg:.4f}")

        if args.l_orth_wv:
            orth_loss_avg = orth_loss_sum / num_batches
            epoch_stats["Avg Orthogonality Loss"] = round(orth_loss_avg, 4)
            logger.info(f"  Avg Orthogonality Loss: {orth_loss_avg:.4f}")

        if use_spatial:
            if spatial_match_count > 0:
                match_avg = spatial_match_sum / spatial_match_count
                epoch_stats["Avg Spatial Match Rate"] = round(match_avg, 4)
                logger.info(f"  Avg Spatial Match Rate: {match_avg:.4f}")
            if args.beta_mix_coef > 0:
                m2_loss_avg = m2_loss_sum / num_batches
                epoch_stats["Avg Scenario-beta m2 Loss"] = round(m2_loss_avg, 4)
                logger.info(f"  Avg Scenario-beta m2 Loss (weighted): {m2_loss_avg:.4f}")

        if args.distil_coef and kd_module is None:
            # Only log basic distillation loss if kd_module is not used
            dist_loss_avg = dist_loss_sum / num_batches
            epoch_stats["Avg Distillation Loss"] = round(dist_loss_avg, 4)
            logger.info(f"  Avg Distillation Loss: {dist_loss_avg:.4f}")
            
            # Add teacher statistics summary
            if teacher_stats_count > 0:
                logger.info(f"  Teacher Model Statistics (averaged over {teacher_stats_count} batches):")
                for key, value in teacher_stats_accumulator.items():
                    avg_value = value / teacher_stats_count
                    epoch_stats[f"Avg Teacher {key}"] = round(avg_value, 4)
                    
                    # Log key metrics to console
                    if key in ['teacher_avg_entropy_combined', 'teacher_avg_confidence_combined', 
                              'teacher_gt_prob_combined', 'teacher_accuracy_combined',
                              'teacher_student_agreement_combined', 'teacher_student_kl_combined',
                              'distillation_temperature', 'teacher_effective_temp_vs_distill_temp']:
                        logger.info(f"    {key}: {avg_value:.4f}")
        
        # Log KD losses summary
        if kd_module is not None:
            has_kd_losses = any(loss_sum > 0 for loss_sum in kd_loss_sums.values())
            if has_kd_losses:
                logger.info(f"  Knowledge Distillation Losses:")
                for loss_name, loss_sum in kd_loss_sums.items():
                    if loss_sum > 0:
                        loss_avg = loss_sum / num_batches
                        epoch_stats[f"Avg KD {loss_name.replace('_', ' ').title()}"] = round(loss_avg, 4)
                        
                        # Log key KD losses to console
                        if loss_name in ['total_kd_loss', 'crd_loss', 'fd_loss', 'icl_loss']:
                            logger.info(f"    {loss_name.replace('_', ' ').title()}: {loss_avg:.4f}")

        # Log final learning rate for the epoch
        final_lr = optimizer.param_groups[0]['lr']
        epoch_stats["Final LR"] = final_lr
        logger.info(f"  Final Learning Rate: {final_lr:.6f}")

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        # Saving model
        if args.save is not None: # and epoch % 9 == 0:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f"checkpoint_{epoch+1}.pt")
            logger.info("Saving model to" + str(model_path))
            model.module.save(model_path)

            #! save the EMA teacher
            if args.distil_coef:
                ema_model_path = os.path.join(args.save, f"checkpoint_{epoch+1}_EMA.pt")
                logger.info("Saving model to" + str(ema_model_path))
                teacher_enc.save(ema_model_path)

            optim_path = os.path.join(args.save, f"optim_{epoch+1}.pt")
            torch.save(optimizer.state_dict(), optim_path)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if fp16_scaler is not None else torch.float32), torch.no_grad():
            evaluate(model, args, classification_head_new, epoch_stats, logger)

        ood_acc, ood_f1 = 0, 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if "Accuracy" in k:
                if k == "ImageNet Accuracy":
                    # ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
            if "Macro F1" in k:
                if k == "ImageNet Macro F1" or k == "IWildCamIDVal Macro F1":
                    continue
                ood_f1 += v
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0
        if num_datasets != 0:
            ood_f1 = ood_f1 / num_datasets
        else:
            ood_f1 = 0
        epoch_stats["Avg OOD Acc"] = round(ood_acc, 4)
        epoch_stats["Avg OOD F1"] = round(ood_f1, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg OOD F1 : {ood_f1:.4f}")
        
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

    if args.save is not None:
        return model_path
