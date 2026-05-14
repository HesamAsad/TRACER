# Alignment Notes: checkpoint_arch_analysis.py vs compare_models_and_gradcam.py

This document notes the key alignments made to ensure `checkpoint_arch_analysis.py` follows the same patterns as the working `compare_models_and_gradcam.py`.

## Key Changes Made

### 1. Model Loading Pattern
**Before:** Custom Args class with minimal fields  
**After:** Uses `build_min_args()` function matching `compare_models_and_gradcam.py`

```python
# Now matches compare_models_and_gradcam.py
args = build_min_args(
    model=model_name_for_clip,
    device=self.device,
    dataset="ImageNet",
    template="openai_imagenet_template"
)
clip_encoder = CLIPEncoder(args, keep_lang=True)
```

### 2. Checkpoint Loading
**Before:** Direct `torch_load()` with custom handling  
**After:** Matches `compare_models_and_gradcam.py` pattern: instantiate then load

```python
# Matches compare_models_and_gradcam.py: load_finetuned_encoder()
enc = CLIPEncoder(args, keep_lang=True)
enc = enc.load(checkpoint_path)
enc = enc.to(self.device)
```

### 3. Device Handling
**Before:** Used `torch.device()` objects  
**After:** Uses device strings (e.g., "cuda", "cuda:0") matching the repo pattern

```python
# Matches compare_models_and_gradcam.py
self.device = config.device if torch.cuda.is_available() else "cpu"
# Can use .to(device) with string
```

### 4. Dataset Loading
**Before:** Direct `torchvision.datasets.ImageFolder`  
**After:** Uses repo's dataset classes via `getattr(datasets, dataset_name)`

```python
# Matches compare_models_and_gradcam.py: get_dataset()
dataset_cls = getattr(datasets, repo_dataset_name, None)
dataset_obj = dataset_cls(
    self.preprocess_fn,
    location=self.data_location,
    batch_size=self.config.batch_size,
    num_workers=self.config.num_workers
)
```

### 5. Zero-shot Head Construction
**Before:** Minimal Args class  
**After:** Uses `build_min_args()` with all required fields (template, train_dataset, data_location)

```python
# Matches compare_models_and_gradcam.py: build_zeroshot_head()
args = build_min_args(
    model="ViT-B-16",
    device=str(self.device),
    dataset="ImageNet",
    template="openai_imagenet_template",
    data_location=os.path.expanduser('~/data')
)
classification_head = get_zeroshot_classifier(args, model.model)
```

### 6. Model Name Normalization
**Consistent:** Both handle ViT-B/16 → ViT-B-16 conversion for open_clip

```python
# Both handle this normalization
if model_name == 'ViT-B/16':
    model_name_for_clip = 'ViT-B-16'  # open_clip uses hyphen
```

## Verified Patterns

✅ Model instantiation: `CLIPEncoder(args, keep_lang=True)`  
✅ Checkpoint loading: `enc.load(checkpoint_path)` instance method  
✅ Device usage: String format ("cuda:0") compatible with `.to(device)`  
✅ Dataset access: Uses repo's dataset classes from `src.datasets_`  
✅ Zero-shot head: Requires full args with template, train_dataset, data_location  
✅ Preprocess access: Uses `model.val_preprocess` from CLIPEncoder  

## State Dict Handling

The state dict extraction keeps the `model.` prefix since:
- All three models (pretrained, direct, pomp) are CLIPEncoder instances
- CLIPEncoder wraps `self.model`, so keys have `model.` prefix
- Consistency is maintained across all comparisons
- Only `module.` prefix (from DataParallel) is stripped

## Remaining Differences (Intentional)

1. **Analysis-specific code**: Parameter metrics, CKA, interpolation - these are new features not in compare_models_and_gradcam.py
2. **Visualization**: Custom plotting functions for analysis outputs
3. **Report generation**: Markdown/PDF report generation for scientific documentation

These differences are intentional and don't affect model loading/checkpoint handling compatibility.

