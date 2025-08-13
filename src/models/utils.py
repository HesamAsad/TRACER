import os

import torch
import pickle
from tqdm import tqdm
import copy
import glob
import numpy as np


def fix_batchnorm_dtype_for_mixed_precision(model):
    """
    Ensure BatchNorm buffers are in float32 for mixed precision training compatibility.
    This fixes the "Expected running_mean to have type Float but got BFloat16" error.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                module.running_mean = module.running_mean.float()
            if hasattr(module, 'running_var') and module.running_var is not None:
                module.running_var = module.running_var.float()
    return model


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None, None, None, 0
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        return None, None, None, 0
    
    # Extract epoch numbers and find the latest
    epochs = []
    for f in checkpoint_files:
        try:
            epoch_num = int(f.split("checkpoint_")[1].split(".pt")[0])
            epochs.append(epoch_num)
        except:
            continue
    
    if not epochs:
        return None, None, None, 0
    
    latest_epoch = max(epochs)
    
    # Construct paths for model, EMA, and optimizer
    model_path = os.path.join(checkpoint_dir, f"checkpoint_{latest_epoch}.pt")
    ema_path = os.path.join(checkpoint_dir, f"checkpoint_{latest_epoch}_EMA.pt")
    optim_path = os.path.join(checkpoint_dir, f"optim_{latest_epoch}.pt")
    
    # Check if files exist
    model_path = model_path if os.path.exists(model_path) else None
    ema_path = ema_path if os.path.exists(ema_path) else None
    optim_path = optim_path if os.path.exists(optim_path) else None
    
    return model_path, ema_path, optim_path, latest_epoch


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def cosine_grad_norm_scheduler(initial_norm, final_norm, steps):
    """
    Cosine scheduler for gradient clipping norm - monotonically increasing.
    
    Args:
        initial_norm: Starting gradient norm value (e.g., 0.0001)
        final_norm: Final gradient norm value (e.g., 0.001) 
        steps: Total number of training steps
    
    Returns:
        Function that takes step and returns current grad norm value
    """
    def _grad_norm_adjuster(step):
        # Ensure step doesn't exceed total steps
        step = min(step, steps - 1)
        
        # Progress from 0 to 1
        progress = step / (steps - 1) if steps > 1 else 0
        
        # Cosine-based smooth increase: starts slow, accelerates, then slows down
        # This creates a smooth S-curve from initial_norm to final_norm
        cosine_factor = 0.5 * (1 - np.cos(np.pi * progress))  # 0 to 1
        
        return initial_norm + (final_norm - initial_norm) * cosine_factor
    
    return _grad_norm_adjuster


def apply_layer_freezing(model, args, logger):
    """
    Apply layer freezing based on arguments.
    Handles all CLIP components: embeddings, transformers, projections, and pooling layers.
    
    Args:
        model: The CLIP model
        args: Arguments containing freeze_text_encoder and trainable_layers
        logger: Logger for reporting
    """
    
    # 1. Freeze text encoder if requested (entire text encoder)
    if args.freeze_text_encoder:
        logger.info("Freezing entire text encoder")
        # Freeze all text-related components
        freeze_components = [
            ('transformer', 'Transformer layers'),
            ('token_embedding', 'Token embeddings'),
            ('positional_embedding', 'Positional embeddings'),
            ('ln_final', 'Final layer norm'),
            ('text_projection', 'Text projection')
        ]
        
        for component_name, description in freeze_components:
            if hasattr(model.model, component_name):
                component = getattr(model.model, component_name)
                if hasattr(component, 'parameters'):
                    for param in component.parameters():
                        param.requires_grad = False
                elif isinstance(component, torch.nn.Parameter):
                    component.requires_grad = False
                logger.info(f"  - Froze {description}")
    else:
        logger.info("Text encoder remains trainable")
    
    # 2. Apply selective layer freezing if specified
    if args.trainable_layers >= 0:
        logger.info(f"Applying selective layer freezing - keeping last {args.trainable_layers} layers trainable")
        
        # Handle text encoder layers (if not completely frozen)
        if not args.freeze_text_encoder and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'resblocks'):
            text_layers = model.model.transformer.resblocks
            total_text_layers = len(text_layers)
            
            if args.trainable_layers == 0:
                # Freeze entire text encoder when trainable_layers=0
                freeze_components = [
                    ('transformer', 'Transformer layers'),
                    ('token_embedding', 'Token embeddings'),
                    ('positional_embedding', 'Positional embeddings'), 
                    ('ln_final', 'Final layer norm'),
                    ('text_projection', 'Text projection')
                ]
                
                for component_name, description in freeze_components:
                    if hasattr(model.model, component_name):
                        component = getattr(model.model, component_name)
                        if hasattr(component, 'parameters'):
                            for param in component.parameters():
                                param.requires_grad = False
                        elif isinstance(component, torch.nn.Parameter):
                            component.requires_grad = False
                logger.info(f"Froze entire text encoder ({total_text_layers} transformer layers + embeddings + projections)")
                
            elif args.trainable_layers < total_text_layers:
                # Freeze first layers, keep last N trainable + keep embeddings/projections trainable
                freeze_until = total_text_layers - args.trainable_layers
                for i, layer in enumerate(text_layers):
                    if i < freeze_until:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(f"Froze first {freeze_until} text transformer layers, keeping last {args.trainable_layers} + embeddings/projections trainable")
            else:
                logger.info(f"All {total_text_layers} text layers remain trainable (requested {args.trainable_layers} >= total)")
        
        # Handle vision encoder
        if hasattr(model.model, 'visual'):
            visual_model = model.model.visual
            
            # Check if it's a Vision Transformer
            if hasattr(visual_model, 'transformer') and hasattr(visual_model.transformer, 'resblocks'):
                # ViT case
                vision_layers = visual_model.transformer.resblocks
                total_vision_layers = len(vision_layers)
                
                if args.trainable_layers == 0:
                    # Freeze entire vision encoder
                    vit_components = [
                        ('conv1', 'Patch embedding conv'),
                        ('class_embedding', 'Class token embedding'),
                        ('positional_embedding', 'Positional embeddings'),
                        ('ln_pre', 'Pre-transformer layer norm'),
                        ('transformer', 'Vision transformer layers'),
                        ('ln_post', 'Post-transformer layer norm'),
                        ('proj', 'Vision projection')
                    ]
                    
                    for component_name, description in vit_components:
                        if hasattr(visual_model, component_name):
                            component = getattr(visual_model, component_name)
                            if hasattr(component, 'parameters'):
                                for param in component.parameters():
                                    param.requires_grad = False
                            elif isinstance(component, torch.nn.Parameter):
                                component.requires_grad = False
                    logger.info(f"Froze entire ViT encoder ({total_vision_layers} transformer layers + embeddings + projections)")
                    
                elif args.trainable_layers < total_vision_layers:
                    # Freeze first layers, keep last N trainable + keep embeddings/projections trainable
                    freeze_until = total_vision_layers - args.trainable_layers
                    for i, layer in enumerate(vision_layers):
                        if i < freeze_until:
                            for param in layer.parameters():
                                param.requires_grad = False
                    logger.info(f"Froze first {freeze_until} ViT layers, keeping last {args.trainable_layers} + embeddings/projections trainable")
                else:
                    logger.info(f"All {total_vision_layers} ViT layers remain trainable (requested {args.trainable_layers} >= total)")
            
            # Check if it's a ResNet
            elif hasattr(visual_model, 'layer1'):
                # ResNet case - we have layer1, layer2, layer3, layer4
                resnet_layers = [visual_model.layer1, visual_model.layer2, visual_model.layer3, visual_model.layer4]
                total_resnet_layers = len(resnet_layers)
                
                if args.trainable_layers == 0:
                    # Freeze entire ResNet encoder
                    resnet_components = [
                        # Stem layers
                        ('conv1', 'Stem conv1'), ('bn1', 'Stem bn1'),
                        ('conv2', 'Stem conv2'), ('bn2', 'Stem bn2'), 
                        ('conv3', 'Stem conv3'), ('bn3', 'Stem bn3'),
                        ('avgpool', 'Stem avgpool'),
                        # Main layers
                        ('layer1', 'ResNet layer1'), ('layer2', 'ResNet layer2'),
                        ('layer3', 'ResNet layer3'), ('layer4', 'ResNet layer4'),
                        # Attention pooling
                        ('attnpool', 'Attention pooling')
                    ]
                    
                    for component_name, description in resnet_components:
                        if hasattr(visual_model, component_name):
                            component = getattr(visual_model, component_name)
                            if hasattr(component, 'parameters'):
                                for param in component.parameters():
                                    param.requires_grad = False
                    logger.info(f"Froze entire ResNet encoder (stem + {total_resnet_layers} layers + attention pooling)")
                    
                elif args.trainable_layers < total_resnet_layers:
                    # Freeze first layers, keep last N trainable + keep stem/pooling trainable
                    freeze_until = total_resnet_layers - args.trainable_layers
                    for i, layer in enumerate(resnet_layers):
                        if i < freeze_until:
                            for param in layer.parameters():
                                param.requires_grad = False
                    logger.info(f"Froze first {freeze_until} ResNet layers, keeping last {args.trainable_layers} + stem/pooling trainable")
                else:
                    logger.info(f"All {total_resnet_layers} ResNet layers remain trainable (requested {args.trainable_layers} >= total)")
    else:
        logger.info("No selective layer freezing applied - all layers remain trainable (except text encoder if frozen)")
    
    # 3. Always keep logit_scale trainable (it's a global parameter)
    if hasattr(model.model, 'logit_scale'):
        model.model.logit_scale.requires_grad = True
        logger.info("Logit scale remains trainable")
    
    # Report final parameter counts with detailed breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Parameter summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    # Detailed breakdown by component
    if hasattr(model.model, 'transformer'):
        text_params = sum(p.numel() for p in model.model.transformer.parameters())
        text_trainable = sum(p.numel() for p in model.model.transformer.parameters() if p.requires_grad)
        logger.info(f"  Text transformer: {text_trainable:,}/{text_params:,} trainable ({100*text_trainable/text_params:.1f}%)")
    
    if hasattr(model.model, 'visual'):
        visual_params = sum(p.numel() for p in model.model.visual.parameters())
        visual_trainable = sum(p.numel() for p in model.model.visual.parameters() if p.requires_grad)
        logger.info(f"  Visual encoder: {visual_trainable:,}/{visual_params:,} trainable ({100*visual_trainable/visual_params:.1f}%)")


# def linear_schedule(base_value, final_value, cur_iter, tot_iter):
#     '''
#     - cur_iter : start from 0
#     '''
#     schedule = ((final_value - base_value) / tot_iter) * cur_iter + base_value
#     return schedule


def accuracy(output, target, topk=(1, )):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0,
                                                  keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(classifier, save_path):
    classifier = copy.deepcopy(classifier)
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier, classification_head=None):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
        if classification_head is None:
            return classifier(inputs)
        classification_head = classification_head.to(inputs.device)
    feats = classifier(inputs)
    return classification_head(feats)


def get_feats(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    feats = classifier(inputs)
    # feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CLIP_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(CLIP_STD).view(3, 1, 1).cuda()

upper_limit, lower_limit = 1, 0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = (X - mu) / std
    return X

def attack_pgd(model, 
               criterion, 
               X, 
               target, 
               alpha=1/255,
               attack_iters=2, 
               norm='l_inf', 
               epsilon=1/255,
               flag='ce'):
    assert flag in ['ce','cl']
    devices = list(range(torch.cuda.device_count()))

    #import pdb; pdb.set_trace()
    
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        attack_imgs = clip_img_preprocessing(X + delta)
        
        if flag == 'ce':
            output = model(attack_imgs)
            loss = criterion(output, target)
        else:
            img_feat, txt_feat, logit_scale2 = model(attack_imgs, target)
            lscale = logit_scale2 if len(devices) == 1 else logit_scale2[0]

            loss, _, _ = criterion(img_feat, txt_feat, lscale)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta