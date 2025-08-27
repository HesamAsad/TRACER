import os
import sys
import argparse
from types import SimpleNamespace
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure repo root is on PYTHONPATH when running directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.datasets_ as datasets
from src.models.modeling import CLIPEncoder
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.common import maybe_dictionarize
from src.models.utils import get_logits
import clip.clip as clip


def build_min_args(model: str,
                   device: str,
                   dataset: str,
                   template: str,
                   batch_size: int,
                   workers: int,
                   data_location: str,
                   use_fp16: int = 1,
                   seed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        device=device,
        train_dataset=dataset,
        template=template,
        batch_size=batch_size,
        workers=workers,
        data_location=data_location,
        use_fp16=use_fp16,
        seed=seed
    )


def load_zero_shot_encoder(args_ns: SimpleNamespace) -> CLIPEncoder:
    enc = CLIPEncoder(args_ns, keep_lang=True)
    return enc


def load_finetuned_encoder(args_ns: SimpleNamespace, checkpoint_path: str, device: str) -> CLIPEncoder:
    # Mirror carot_loss style: instantiate then load via instance method
    enc = CLIPEncoder(args_ns, keep_lang=True)
    enc = enc.load(checkpoint_path)
    enc = enc.to(device)
    return enc


def get_dataset(preprocess, dataset_name: str, data_location: str, batch_size: int, workers: int):
    dataset_cls = getattr(datasets, dataset_name)
    ds = dataset_cls(preprocess, location=data_location, batch_size=batch_size, num_workers=workers)
    return ds


def build_zeroshot_head(args_ns: SimpleNamespace, clip_model) -> torch.nn.Module:
    return get_zeroshot_classifier(args_ns, clip_model)


def tokenize_templates_for_classnames(classnames: List[str], template_fns: List, device: str) -> torch.Tensor:
    texts = []
    for classname in classnames:
        prompts = [t(classname) for t in template_fns]
        texts.append(clip.tokenize(prompts))
    token_batch = torch.cat(texts, dim=0).to(device)
    return token_batch


class GradCAM_CLIP:
    """
    Minimal Grad-CAM for CLIP visual encoders (supports ViT and ResNet backbones in this repo).
    - ViT: uses gradients wrt patch embedding map from visual.conv1 output (grid-level CAM).
    - ResNet: uses gradients wrt last bottleneck conv (layer4[-1].conv3) for high-level CAM.
    """
    def __init__(self, clip_encoder: CLIPEncoder, target_layer: torch.nn.Module):
        self.encoder = clip_encoder
        self.model = clip_encoder.model  # underlying CLIP
        self.visual = self.model.visual
        self.device = next(self.model.parameters()).device
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # register hooks
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output
        # ensure we can take grad on activations
        if isinstance(self.activations, torch.Tensor) and self.activations.requires_grad:
            self.activations.retain_grad()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def remove_hooks(self):
        if hasattr(self, '_fwd_handle') and self._fwd_handle is not None:
            self._fwd_handle.remove()
        if hasattr(self, '_bwd_handle') and self._bwd_handle is not None:
            self._bwd_handle.remove()

    def __del__(self):
        self.remove_hooks()

    def compute_cam(self, images: torch.Tensor, logits: torch.Tensor, target_indices: torch.Tensor) -> np.ndarray:
        """
        images: [B,3,H,W] preprocessed
        logits: [B,C]
        target_indices: [B]
        returns: np array of shape [B, H, W] normalized 0..1
        """
        # Backpropagate on target class score
        self.encoder.zero_grad()
        self.model.zero_grad()
        selected = logits.gather(1, target_indices[:, None]).sum()
        selected.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients if self.gradients is not None else (acts.grad if hasattr(acts, 'grad') else None)
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Check target layer selection.")

        # Global average pooling over spatial dims
        if acts.dim() == 4:  # [B,C,H,W]
            weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
            cam = (weights * acts).sum(dim=1)  # [B,H,W]
            cam = F.relu(cam)
            cams = []
            for i in range(cam.shape[0]):
                m = cam[i]
                m -= m.min()
                if m.max() > 0:
                    m = m / (m.max() + 1e-6)
                cams.append(m)
            cam = torch.stack(cams, dim=0)
            # Upsample to input spatial size
            cam = F.interpolate(cam[:, None, :, :], size=images.shape[-2:], mode='bilinear', align_corners=False)[:, 0, :, :]
        elif acts.dim() == 3:  # [B,Tokens,Channels] (unlikely here)
            # fallback: average channel gradients, map tokens (excluding CLS) to grid
            B, T, C = acts.shape
            grads_mean = grads.mean(dim=2)  # [B,T]
            # exclude CLS token assumed at index 0; infer grid size
            token_scores = grads_mean[:, 1:]
            S = int((token_scores.shape[1]) ** 0.5)
            token_scores = token_scores.reshape(B, 1, S, S)
            token_scores = F.relu(token_scores)
            token_scores = F.interpolate(token_scores, size=images.shape[-2:], mode='bilinear', align_corners=False)
            cam = token_scores[:, 0]
        else:
            raise RuntimeError("Unsupported activation shape for CAM.")

        cam_np = cam.detach().float().cpu().numpy()
        # release references to free graph memory for next call
        self.activations = None
        self.gradients = None
        return cam_np


def pick_target_layer_for_cam(visual) -> torch.nn.Module:
    # ResNet-like backbone
    if hasattr(visual, 'attnpool') and hasattr(visual, 'layer4'):
        last_block = visual.layer4[-1]
        if hasattr(last_block, 'conv3'):
            return last_block.conv3
        return visual.layer4  # fallback
    # ViT-like backbone
    if hasattr(visual, 'transformer') and hasattr(visual, 'conv1'):
        return visual.conv1  # patch embedding map
    # Fallback
    raise RuntimeError("Unsupported CLIP visual backbone for Grad-CAM.")


def overlay_cam_on_image(cam: np.ndarray, image: Image.Image, alpha: float = 0.35) -> Image.Image:
    # image is PIL (un-normalized). Convert cam to heatmap and overlay
    import matplotlib.cm as cm
    heatmap = (cm.jet(cam)[..., :3] * 255.0).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BICUBIC)
    return Image.blend(image.convert('RGB'), heatmap, alpha)


def denormalize_clip_tensor(t: torch.Tensor) -> Image.Image:
    # t: [3,H,W] normalized by CLIP mean/std
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=t.device)[:, None, None]
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=t.device)[:, None, None]
    img = t * CLIP_STD + CLIP_MEAN
    img = (img.clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(img)


def compute_alignment_and_uniformity(clip_encoder: CLIPEncoder,
                                     dataset,
                                     template_name: str,
                                     device: str,
                                     max_batches: int = 50) -> Dict[str, float]:
    import src.templates as templates
    template_fns = getattr(templates, template_name)
    classnames = dataset.classnames

    with torch.no_grad():
        # Precompute per-class text embeddings (normalized)
        text_embeds = []
        for classname in classnames:
            texts = [t(classname) for t in template_fns]
            texts = clip.tokenize(texts).to(device)
            emb = clip_encoder.model.encode_text(texts)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            emb = emb / emb.norm()
            text_embeds.append(emb)
        text_embeds = torch.stack(text_embeds, dim=0)  # [C, D]

    # Iterate images and compute image embeddings
    dataloader = dataset.test_loader
    image_feats = []
    label_text_feats = []
    count = 0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = maybe_dictionarize(batch)
        x = batch['images'].to(device)
        y = batch['labels'].to(device)
        with torch.no_grad():
            img_feat = clip_encoder.model.encode_image(x)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        image_feats.append(img_feat)
        label_text_feats.append(text_embeds[y].to(device))
        count += 1
        if max_batches is not None and count >= max_batches:
            break

    image_feats = torch.cat(image_feats, dim=0)  # [N,D]
    label_text_feats = torch.cat(label_text_feats, dim=0)  # [N,D]

    # Alignment: E[||x - y||^2]
    alignment = (image_feats - label_text_feats).pow(2).sum(dim=1).mean().item()

    # Uniformity: log E_{i!=j} exp(-t * ||x_i - x_j||^2)
    def uniformity(feats: torch.Tensor, t: float = 2.0, sample_pairs: int = 20000) -> float:
        n = feats.shape[0]
        if n < 2:
            return float('nan')
        # random pairs
        idx_i = torch.randint(0, n, (sample_pairs,), device=feats.device)
        idx_j = torch.randint(0, n, (sample_pairs,), device=feats.device)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        diffs = feats[idx_i] - feats[idx_j]
        dist2 = (diffs * diffs).sum(dim=1)
        val = torch.exp(-t * dist2).mean().log().item()
        return val

    uniformity_img = uniformity(image_feats)
    uniformity_txt = uniformity(label_text_feats)

    return {
        'alignment_euclidean_sq': alignment,
        'uniformity_image_log': uniformity_img,
        'uniformity_text_log': uniformity_txt,
        'num_samples': image_feats.shape[0]
    }


def main():
    parser = argparse.ArgumentParser(description="Compare zero-shot and fine-tuned CLIP encoders: Grad-CAM + alignment/uniformity")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to finetuned CLIPEncoder checkpoint_*.pt')
    parser.add_argument('--clip_load', type=str, default=None, help='Alias for --checkpoint to mirror carot_loss')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset class name from src.datasets_ (e.g., ImageNet, ImageNetV2, ImageNetR, ImageNetA, ImageNetSketch, ObjectNet)')
    parser.add_argument('--data-location', type=str, required=True, help='Root path containing datasets (same as training)')
    parser.add_argument('--model', type=str, default='ViT-L/14', help='CLIP model name (must match training, e.g. ViT-L/14)')
    parser.add_argument('--template', type=str, default='openai_imagenet_template', help='Prompt template name in src.templates')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--use-fp16', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--viz-samples', type=int, default=10000, help='Number of images to visualize with Grad-CAM')
    parser.add_argument('--metrics-batches', type=int, default=500, help='Batches to use for alignment/uniformity (None=all)')
    parser.add_argument('--out-dir', type=str, default='./compare_outputs')

    args_local = parser.parse_args()

    os.makedirs(args_local.out_dir, exist_ok=True)

    # Build minimal args used across helpers
    zs_args = build_min_args(
        model=args_local.model,
        device=args_local.device,
        dataset=args_local.dataset,
        template=args_local.template,
        batch_size=args_local.batch_size,
        workers=args_local.workers,
        data_location=args_local.data_location,
        use_fp16=args_local.use_fp16,
        seed=0,
    )

    # Load encoders
    print('Loading zero-shot encoder...')
    zs_enc = load_zero_shot_encoder(zs_args).to(args_local.device)
    # Resolve checkpoint path (support both flags)
    ckpt_path = args_local.clip_load or args_local.checkpoint
    if ckpt_path is None:
        raise ValueError('Please provide --checkpoint or --clip_load path to a finetuned CLIPEncoder checkpoint.')

    print('Loading finetuned encoder...')
    ft_enc = load_finetuned_encoder(zs_args, ckpt_path, args_local.device)

    # Dataset (use finetuned preprocess for loading to match training/eval)
    print(f'Preparing dataset {args_local.dataset} ...')
    dataset = get_dataset(ft_enc.val_preprocess, args_local.dataset, args_local.data_location, args_local.batch_size, args_local.workers)
    dataloader = dataset.train_loader

    # Build zero-shot heads (consistent with carot evaluation)
    print('Building zero-shot classification heads...')
    zs_head_for_zs = build_zeroshot_head(zs_args, zs_enc.model).to(args_local.device)
    zs_head_for_ft = build_zeroshot_head(zs_args, ft_enc.model).to(args_local.device)

    # Prepare Grad-CAM modules
    print('Preparing Grad-CAM...')
    zs_target_layer = pick_target_layer_for_cam(zs_enc.model.visual)
    ft_target_layer = pick_target_layer_for_cam(ft_enc.model.visual)
    zs_cam = GradCAM_CLIP(zs_enc, zs_target_layer)
    ft_cam = GradCAM_CLIP(ft_enc, ft_target_layer)

    # Visualization loop
    print('Running Grad-CAM visualizations...')
    saved = 0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = maybe_dictionarize(batch)
        images = batch['images']
        labels = batch['labels']  # keep on CPU for labeling
        images_cpu = images.detach().cpu()
        batch_size = images_cpu.shape[0]

        cam_chunk = max(1, min(8, batch_size))  # small chunks to reduce peak memory
        j = 0
        while j < batch_size:
            if saved >= args_local.viz_samples:
                break
            j_end = min(batch_size, j + cam_chunk)
            imgs_chunk_cpu = images_cpu[j:j_end]

            # Zero-shot CAMs for chunk
            imgs_chunk = imgs_chunk_cpu.to(args_local.device, dtype=torch.bfloat16 if args_local.use_fp16 else torch.float32, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args_local.use_fp16 else torch.float32):
                logits_zs = get_logits(imgs_chunk, zs_enc, zs_head_for_zs)
            pred_zs = logits_zs.argmax(dim=1)
            cams_zs = zs_cam.compute_cam(imgs_chunk, logits_zs, pred_zs)
            pred_zs_cpu = pred_zs.detach().cpu().tolist()
            zs_enc.zero_grad(set_to_none=True)
            del logits_zs, pred_zs, imgs_chunk
            torch.cuda.empty_cache()

            # Fine-tuned CAMs for chunk
            imgs_chunk = imgs_chunk_cpu.to(args_local.device, dtype=torch.bfloat16 if args_local.use_fp16 else torch.float32, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args_local.use_fp16 else torch.float32):
                logits_ft = get_logits(imgs_chunk, ft_enc, zs_head_for_ft)
            pred_ft = logits_ft.argmax(dim=1)
            cams_ft = ft_cam.compute_cam(imgs_chunk, logits_ft, pred_ft)
            pred_ft_cpu = pred_ft.detach().cpu().tolist()
            ft_enc.zero_grad(set_to_none=True)
            del logits_ft, pred_ft, imgs_chunk
            torch.cuda.empty_cache()

            # Save overlays for chunk
            for k in range(j, j_end):
                idx_local = k - j
                img_pil = denormalize_clip_tensor(imgs_chunk_cpu[idx_local])
                over_zs = overlay_cam_on_image(cams_zs[idx_local], img_pil, alpha=0.35)
                over_ft = overlay_cam_on_image(cams_ft[idx_local], img_pil, alpha=0.35)

                zs_cls = int(pred_zs_cpu[idx_local]) if idx_local < len(pred_zs_cpu) else -1
                ft_cls = int(pred_ft_cpu[idx_local]) if idx_local < len(pred_ft_cpu) else -1
                fname = f"{args_local.dataset}_idx{i*args_local.batch_size+k}_gt{int(labels[k])}_zs{zs_cls}_ft{ft_cls}.png"
                canvas = Image.new('RGB', (img_pil.width * 3, img_pil.height))
                canvas.paste(img_pil, (0, 0))
                canvas.paste(over_zs, (img_pil.width, 0))
                canvas.paste(over_ft, (img_pil.width * 2, 0))
                canvas.save(os.path.join(args_local.out_dir, fname))

                saved += 1
                if saved >= args_local.viz_samples:
                    break

            # free per-chunk CPU arrays
            del cams_zs, cams_ft, imgs_chunk_cpu
            torch.cuda.empty_cache()
            j = j_end

        if saved >= args_local.viz_samples:
            break

    # Clean up CAM hooks before metrics to avoid interfering with no_grad paths
    zs_cam.remove_hooks()
    ft_cam.remove_hooks()

    # Metrics: alignment & uniformity
    print('Computing alignment and uniformity metrics (zero-shot vs finetuned encoders)...')
    zs_metrics = compute_alignment_and_uniformity(zs_enc, dataset, args_local.template, args_local.device, max_batches=args_local.metrics_batches)
    ft_metrics = compute_alignment_and_uniformity(ft_enc, dataset, args_local.template, args_local.device, max_batches=args_local.metrics_batches)

    # Save metrics
    import json
    with open(os.path.join(args_local.out_dir, 'metrics.json'), 'w') as f:
        json.dump({'zero_shot': zs_metrics, 'finetuned': ft_metrics}, f, indent=2)

    print('Done. Outputs saved to:', args_local.out_dir)


if __name__ == '__main__':
    main()


