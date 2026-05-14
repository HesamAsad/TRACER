import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SD_WEIGHT = 0.1

TEXT_TEMPLATES = [
            "the digit {}", 
            "number {}", 
            "handwritten {}", 
            "a {} digit",
            "the number {} written by hand"
        ]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure autocast device type and dtype (safer defaults across hardware)
autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
if autocast_device == 'cuda':
    try:
        bf16_ok = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    autocast_dtype = torch.bfloat16 if bf16_ok else torch.float16
else:
    # CPU autocast prefers bfloat16
    autocast_dtype = torch.bfloat16
print(f"Using autocast device: {autocast_device}, dtype: {autocast_dtype}")

# Optional matmul precision for speed on Ampere+ GPUs
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ==================== Data Preparation ====================

class MNISTMultiModal(Dataset):
    """MNIST dataset with generated text descriptions"""
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset
        # Simple text templates for MNIST digits
        self.templates = TEXT_TEMPLATES
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        # Generate text description - select template randomly
        template = random.choice(self.templates)
        text = template.format(label)
        # Convert text to token indices (simple character-level encoding)
        text_tokens = self.text_to_tokens(text, max_len=32)
        return image, text_tokens, label
    
    def text_to_tokens(self, text, max_len=32):
        # Use shared tokenization function
        return tokenize_text(text, max_len)

class ColoredMNISTMultiModal(Dataset):
    """Colored MNIST dataset with spurious correlations between colors and digit classes"""
    def __init__(self, mnist_dataset, color_shift=0, spurious_strength=0.995, invert_mapping=False, color_intensity=3.0):
        self.mnist = mnist_dataset
        self.color_shift = color_shift
        self.spurious_strength = float(spurious_strength)
        self.invert_mapping = bool(invert_mapping)
        self.color_intensity = float(color_intensity)
        # Text templates for colored digits
        self.templates = TEXT_TEMPLATES
        self.colors = ['red', 'blue']
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        
        # Determine color based on spurious correlation
        # Apply a cyclic shift to introduce distribution shift across splits if desired
        shifted_label = (int(label) + int(self.color_shift)) % 10
        # Base mapping: digits 0-4 -> red, 5-9 -> blue
        base_color = 'red' if shifted_label < 5 else 'blue'
        if self.invert_mapping:
            base_color = 'blue' if base_color == 'red' else 'red'
        # With high probability, use base color; otherwise flip to opposite color
        if random.random() < self.spurious_strength:
            color = base_color
        else:
            color = 'blue' if base_color == 'red' else 'red'
        
        # Apply color transformation to the image
        colored_image = self.apply_color(image, color)
        
        # Generate text description with color
        template = random.choice(self.templates)
        text = template.format(label)
        
        # Convert text to token indices
        text_tokens = self.text_to_tokens(text, max_len=32)
        return colored_image, text_tokens, label
    
    def apply_color(self, image, color):
        """Apply strong color transformation to RGB image in unnormalized space, then re-normalize."""
        if image.shape[0] != 3:
            return image

        # Unnormalize to [0, 1]
        mean = torch.tensor([0.1307, 0.1307, 0.1307], dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor([0.3081, 0.3081, 0.3081], dtype=image.dtype, device=image.device).view(3, 1, 1)
        img_unnorm = torch.clamp(image * std + mean, 0.0, 1.0)

        # Stronger tint via multiplicative scaling
        boost = max(self.color_intensity, 1.0)
        reduce = max(0.05, min(1.0 / boost, 0.5))
        if color == 'red':
            multipliers = torch.tensor([boost, reduce, reduce], dtype=img_unnorm.dtype, device=img_unnorm.device).view(3, 1, 1)
        elif color == 'blue':
            multipliers = torch.tensor([reduce, reduce, boost], dtype=img_unnorm.dtype, device=img_unnorm.device).view(3, 1, 1)
        else:
            multipliers = torch.ones(3, 1, 1, dtype=img_unnorm.dtype, device=img_unnorm.device)

        colored = torch.clamp(img_unnorm * multipliers, 0.0, 1.0)
        # Re-normalize
        colored_norm = (colored - mean) / std
        return colored_norm
    
    def text_to_tokens(self, text, max_len=32):
        # Use shared tokenization function
        return tokenize_text(text, max_len)

def tokenize_text(text, max_len=32):
    """
    Shared tokenization function for text.
    
    Args:
        text: Input text string
        max_len: Maximum sequence length
    
    Returns:
        tokens: Tensor of token indices
    """
    # Simple character-level tokenization
    vocab = " 0123456789abcdefghijklmnopqrstuvwxyz{}"
    tokens = [vocab.index(c) if c in vocab else 0 for c in text.lower()]
    # Pad or truncate to max_len
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)

def get_zeroshot_classifier(text_encoder, templates, num_classes=10):
    """
    Create a zero-shot classifier by computing mean embeddings of all text templates.
    
    Args:
        text_encoder: The text encoder model
        templates: List of text templates
        num_classes: Number of classes (digits 0-9)
    
    Returns:
        classifier_weights: Tensor of shape (num_classes, embed_dim) containing mean embeddings
    """
    text_encoder.eval()
    classifier_weights = []
    
    with torch.no_grad():
        for digit in range(num_classes):
            # Get embeddings for all templates for this digit
            digit_embeddings = []
            for template in templates:
                text = template.format(digit)
                # Use shared tokenization function
                tokens = tokenize_text(text, max_len=32).unsqueeze(0).to(device)
                
                # Get embedding with autocast
                with autocast(device_type=autocast_device, dtype=autocast_dtype):
                    embedding = text_encoder(tokens, return_features=True)
                # Normalize each template embedding before averaging for stability
                embedding = F.normalize(embedding, dim=-1)
                
                # Ensure embedding is 2D: (batch_size, embed_dim)
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                digit_embeddings.append(embedding)
            
            # Compute mean embedding for this digit
            # Stack along batch dimension and take mean
            mean_embedding = torch.cat(digit_embeddings, dim=0).mean(dim=0)  # Shape: (embed_dim,)
            mean_embedding = F.normalize(mean_embedding, dim=-1)
            classifier_weights.append(mean_embedding)
    
    return torch.stack(classifier_weights)  # Shape: (num_classes, embed_dim)

def evaluate_zeroshot_classifier(image_encoder, classifier_weights, data_loader):
    """
    Evaluate zero-shot classification performance.
    
    Args:
        image_encoder: The image encoder model
        classifier_weights: Zero-shot classifier weights
        data_loader: Data loader for evaluation
    
    Returns:
        accuracy: Classification accuracy percentage
    """
    image_encoder.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, _, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get image embeddings with autocast
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                image_features = image_encoder(images, return_features=True)
            # Normalize both features and weights to use cosine similarity
            image_features = F.normalize(image_features, dim=-1)
            norm_weights = F.normalize(classifier_weights, dim=-1)
            
            # Compute similarity with classifier weights
            similarity = torch.matmul(image_features, norm_weights.T)
            
            # Get predictions
            _, predicted = torch.max(similarity, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Transform for RGB images (for pre-training)
transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to RGB
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

# Load full dataset
DATA_ROOT = os.getenv('DATA_ROOT', '../data_hesam')
full_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT, train=True, download=True, transform=transform_rgb
)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT, train=False, download=True, transform=transform_rgb
)

# Split training data into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create multimodal datasets for pre-training (original MNIST)
train_mm = MNISTMultiModal(train_dataset)
val_mm = MNISTMultiModal(val_dataset)
test_mm = MNISTMultiModal(test_dataset)

# Create colored MNIST datasets for fine-tuning
# Allow configurable, harsh distribution shift per split via environment variables
COLOR_SHIFT_TRAIN = int(os.getenv("COLOR_SHIFT_TRAIN", "0"))
COLOR_SHIFT_VAL = int(os.getenv("COLOR_SHIFT_VAL", "0"))
COLOR_SHIFT_TEST = int(os.getenv("COLOR_SHIFT_TEST", "0"))
SPURIOUS_STRENGTH_TRAIN = float(os.getenv("SPURIOUS_STRENGTH_TRAIN", "0.95"))
SPURIOUS_STRENGTH_VAL = float(os.getenv("SPURIOUS_STRENGTH_VAL", "0.95"))
SPURIOUS_STRENGTH_TEST = float(os.getenv("SPURIOUS_STRENGTH_TEST", "0.95"))
INVERT_VAL = bool(int(os.getenv("INVERT_MAPPING_VAL", "0")))
INVERT_TEST = bool(int(os.getenv("INVERT_MAPPING_TEST", "0")))
COLOR_INTENSITY = float(os.getenv("COLOR_INTENSITY", "5.0"))

train_colored = ColoredMNISTMultiModal(
    train_dataset,
    color_shift=COLOR_SHIFT_TRAIN,
    spurious_strength=SPURIOUS_STRENGTH_TRAIN,
    invert_mapping=False,
    color_intensity=COLOR_INTENSITY,
)
val_colored = ColoredMNISTMultiModal(
    val_dataset,
    color_shift=COLOR_SHIFT_VAL,
    spurious_strength=SPURIOUS_STRENGTH_VAL,
    invert_mapping=INVERT_VAL,
    color_intensity=COLOR_INTENSITY,
)
test_colored = ColoredMNISTMultiModal(
    test_dataset,
    color_shift=COLOR_SHIFT_TEST,
    spurious_strength=SPURIOUS_STRENGTH_TEST,
    invert_mapping=INVERT_TEST,
    color_intensity=COLOR_INTENSITY,
)

# Create dataloaders
batch_size = int(os.getenv("BATCH_SIZE", "128"))
num_workers = int(os.getenv("NUM_WORKERS", str(min(8, max(2, (os.cpu_count() or 4)//2)))))
pin_memory = bool(torch.cuda.is_available())
train_loader = DataLoader(train_mm, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_mm, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_mm, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Create colored dataloaders for fine-tuning
train_colored_loader = DataLoader(train_colored, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_colored_loader = DataLoader(val_colored, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_colored_loader = DataLoader(test_colored, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ==================== Model Architectures ====================

class LightViT(nn.Module):
    """Lightweight Vision Transformer for MNIST"""
    def __init__(self, img_size=28, patch_size=4, in_channels=3, embed_dim=128, 
                 depth=4, num_heads=4, mlp_ratio=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x, return_features=False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract cls token
        cls_output = x[:, 0]
        
        if return_features:
            return self.projection(cls_output)
        else:
            return self.classifier(cls_output)

class TransformerBlock(nn.Module):
    """Bigger transformer block with more capacity"""
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads * 2, batch_first=True)  # double the heads
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio * 2)  # double the MLP size
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        # Add dropout for regularization
        self.dropout_attn = nn.Dropout(0.1)
        self.dropout_mlp = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out
        
        # MLP
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        mlp_out = self.dropout_mlp(mlp_out)
        x = x + mlp_out

        # Extra normalization for stability
        x = self.norm3(x)
        return x

class LightTextTransformer(nn.Module):
    """Lightweight text transformer"""
    def __init__(self, vocab_size=40, embed_dim=128, depth=2, num_heads=4, 
                 max_len=32, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2)
            for _ in range(depth)
        ])
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, return_features=False):
        # Token embedding
        x = self.token_embed(x)
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        if return_features:
            return self.projection(x)
        else:
            return self.classifier(x)

class MultiModalContrastiveModel(nn.Module):
    """Combined model for contrastive pre-training"""
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, images, texts, return_features=False):
        image_features = self.image_encoder(images, return_features=True)
        text_features = self.text_encoder(texts, return_features=True)
        
        if return_features:
            return image_features, text_features
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features

# ==================== Loss Functions ====================

def contrastive_loss(image_features, text_features, temperature):
    """Linear version of contrastive loss (InfoNCE)"""
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarity = torch.matmul(image_features, text_features.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Compute loss for both directions
    loss_i2t = F.cross_entropy(similarity, labels)
    loss_t2i = F.cross_entropy(similarity.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

# ==================== Training Functions ====================

def pretrain_contrastive(model, train_loader, val_loader, epochs=10, checkpoint_dir="toy_exp_ckpts"):
    """Pre-train the model using contrastive learning"""
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    train_losses = []
    val_losses = []
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, texts, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass with bfloat16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                image_features, text_features = model(images, texts)
                loss = contrastive_loss(image_features, text_features, model.temperature)
            
            # Direct backward pass (no scaling needed for bfloat16)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts, _ in val_loader:
                images, texts = images.to(device), texts.to(device)
                
                # Use autocast for validation too
                with autocast(device_type=autocast_device, dtype=autocast_dtype):
                    image_features, text_features = model(images, texts)
                    loss = contrastive_loss(image_features, text_features, model.temperature)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint for each epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"pretrain_epoch{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pretrained_final.pth"))
    return train_losses, val_losses

def evaluate_classification(model, data_loader, modality='image'):
    """Evaluate classification accuracy using classification heads"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts, labels in data_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                if modality == 'image':
                    outputs = model.image_encoder(images, return_features=False)
                else:
                    outputs = model.text_encoder(texts, return_features=False)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def evaluate_with_mean_embeddings(model, data_loader, templates, modality='image'):
    """Evaluate classification accuracy using mean embeddings approach"""
    model.eval()
    correct = 0
    total = 0
    
    # Create classifier weights using mean embeddings
    if modality == 'image':
        # For image modality, use text encoder to create classifier weights
        classifier_weights = get_zeroshot_classifier(model.text_encoder, templates)
        
        with torch.no_grad():
            for images, _, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Get image embeddings with autocast
                with autocast(device_type=autocast_device, dtype=autocast_dtype):
                    image_features = model.image_encoder(images, return_features=True)
                image_features = F.normalize(image_features, dim=-1)
                norm_weights = F.normalize(classifier_weights, dim=-1)
                
                # Compute similarity with classifier weights
                # image_features: (batch_size, embed_dim)
                # classifier_weights: (num_classes, embed_dim)
                # We need: (batch_size, num_classes)
                similarity = torch.matmul(image_features, norm_weights.T)
                
                # Get predictions
                _, predicted = torch.max(similarity, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    else:
        # For text modality, use image encoder to create classifier weights
        # We need to create "mean image embeddings" for each digit
        # This is more complex, so we'll use a simplified approach
        # Create text embeddings for all templates and digits
        all_text_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for digit in range(10):
                for template in templates:
                    text = template.format(digit)
                    tokens = tokenize_text(text, max_len=32).unsqueeze(0).to(device)
                    
                    with autocast(device_type=autocast_device, dtype=autocast_dtype):
                        embedding = model.text_encoder(tokens, return_features=True)
                    embedding = F.normalize(embedding, dim=-1)
                    
                    all_text_embeddings.append(embedding)
                    all_labels.append(digit)
            
            # Create classifier weights from text embeddings
            text_embeddings = torch.cat(all_text_embeddings, dim=0)
            labels = torch.tensor(all_labels, device=device)
            
            # Compute mean embeddings for each digit
            classifier_weights = []
            for digit in range(10):
                digit_mask = labels == digit
                mean_embedding = text_embeddings[digit_mask].mean(dim=0)
                mean_embedding = F.normalize(mean_embedding, dim=-1)
                classifier_weights.append(mean_embedding)
            classifier_weights = torch.stack(classifier_weights)
            
            # Evaluate
            for _, texts, labels in data_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                # Get text embeddings with autocast
                with autocast(device_type=autocast_device, dtype=autocast_dtype):
                    text_features = model.text_encoder(texts, return_features=True)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarity with classifier weights
                # text_features: (batch_size, embed_dim)
                # classifier_weights: (num_classes, embed_dim)
                # We need: (batch_size, num_classes)
                similarity = torch.matmul(text_features, classifier_weights.T)
                
                # Get predictions
                _, predicted = torch.max(similarity, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# ==================== Fine-tuning Strategies ====================

def finetune_direct(model, train_loader, val_loader, epochs=10, checkpoint_dir="toy_exp_ckpts"):
    """Strategy 1: Direct full fine-tuning with frozen text encoder, using contrastive loss"""
    model_ft = deepcopy(model)
    
    # Freeze text encoder
    for param in model_ft.text_encoder.parameters():
        param.requires_grad = False
    for param in model_ft.image_encoder.parameters():
        param.requires_grad = True

    # Only optimize image encoder
    optimizer = optim.AdamW(model_ft.image_encoder.parameters(), lr=1e-4, weight_decay=0.001)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        model_ft.train()
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass with bfloat16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                image_features = model_ft.image_encoder(images, return_features=True)
                with torch.no_grad():
                    text_features = model_ft.text_encoder(texts, return_features=True)
                loss = contrastive_loss(image_features, text_features, model_ft.temperature)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
        # Save checkpoint for each epoch
        torch.save(model_ft.state_dict(), os.path.join(checkpoint_dir, f"direct_ft_epoch{epoch+1}.pth"))
    # Save final model
    torch.save(model_ft.state_dict(), os.path.join(checkpoint_dir, "direct_ft_final.pth"))
    return model_ft

def finetune_l2_regularization(model, train_loader, val_loader, epochs=10, lambda_reg=1e-2, checkpoint_dir="toy_exp_ckpts"):
    """Strategy 2: Fine-tuning with L2 regularization and frozen text encoder, using contrastive loss"""
    model_ft = deepcopy(model)
    
    # Freeze text encoder
    for param in model_ft.text_encoder.parameters():
        param.requires_grad = False
    
    # Store initial parameters for image encoder only
    initial_params = {name: param.clone() for name, param in model_ft.image_encoder.named_parameters()}
    
    for param in model_ft.image_encoder.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model_ft.image_encoder.parameters(), lr=1e-4, weight_decay=0.001)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        model_ft.train()
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass with bfloat16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                image_features = model_ft.image_encoder(images, return_features=True)
                with torch.no_grad():
                    text_features = model_ft.text_encoder(texts, return_features=True)
                contrastive = contrastive_loss(image_features, text_features, model_ft.temperature)
                
                # L2 regularization loss (only for image encoder)
                reg_loss = 0
                for name, param in model_ft.image_encoder.named_parameters():
                    reg_loss += torch.sum((param - initial_params[name]) ** 2)
                
                loss = contrastive + lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
        # Save checkpoint for each epoch
        torch.save(model_ft.state_dict(), os.path.join(checkpoint_dir, f"l2reg_ft_epoch{epoch+1}.pth"))
    # Save final model
    torch.save(model_ft.state_dict(), os.path.join(checkpoint_dir, "l2reg_ft_final.pth"))
    return model_ft

def finetune_self_distillation(model, train_loader, val_loader, epochs=10, temperature=0.1, checkpoint_dir="toy_exp_ckpts"):
    """Strategy 3: Self-distillation with frozen text encoder, using contrastive loss and KL distillation over probabilities"""
    teacher = deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    
    student = deepcopy(model)
    
    # Freeze text encoder
    for param in student.text_encoder.parameters():
        param.requires_grad = False
    
    for param in student.image_encoder.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(student.image_encoder.parameters(), lr=1e-4, weight_decay=0.001)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        student.train()
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass with bfloat16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                # Student features
                student_img_features = student.image_encoder(images, return_features=True)
                with torch.no_grad():
                    student_txt_features = student.text_encoder(texts, return_features=True)
                contrastive = contrastive_loss(student_img_features, student_txt_features, temperature)
                
                # Teacher features (no gradient)
                with torch.no_grad():
                    teacher_img_features = teacher.image_encoder(images, return_features=True)
                    # Compute logits for both student and teacher
                    # Use the text features as "class prototypes"
                    # (batch, embed_dim) @ (num_classes, embed_dim).T -> (batch, num_classes)
                    # Here, we use the batch's own text features as the "classes"
                    # This is a simplification; for real distillation, use a fixed set of class prototypes
                    teacher_logits = torch.matmul(teacher_img_features, student_txt_features.T)
                    student_logits = torch.matmul(student_img_features, student_txt_features.T)
                    # Softmax over classes (dim=1)
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    student_log_probs = F.log_softmax(student_logits, dim=1)
                # KL divergence (teacher_probs is detached)
                soft_loss_img = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                loss = contrastive + SD_WEIGHT * soft_loss_img
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
        # Save checkpoint for each epoch
        torch.save(student.state_dict(), os.path.join(checkpoint_dir, f"selfdistill_ft_epoch{epoch+1}.pth"))
    # Save final model
    torch.save(student.state_dict(), os.path.join(checkpoint_dir, "selfdistill_ft_final.pth"))
    return student

def finetune_dynamic_self_distillation(model, train_loader, val_loader, epochs=10, 
                                      temperature=0.1, ema_decay=0.99999, checkpoint_dir="toy_exp_ckpts"):
    """Strategy 4: Dynamic self-distillation with frozen text encoder, using contrastive loss and KL distillation over probabilities"""
    student = deepcopy(model)
    teacher = deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    
    # Freeze text encoder
    for param in student.text_encoder.parameters():
        param.requires_grad = False
    
    for param in student.image_encoder.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(student.image_encoder.parameters(), lr=1e-4, weight_decay=0.001)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        student.train()
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass with bfloat16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                # Student features
                student_img_features = student.image_encoder(images, return_features=True)
                with torch.no_grad():
                    student_txt_features = student.text_encoder(texts, return_features=True)
                contrastive = contrastive_loss(student_img_features, student_txt_features, temperature)
                
                # Teacher features (no gradient)
                with torch.no_grad():
                    teacher_img_features = teacher.image_encoder(images, return_features=True)
                    # Compute logits for both student and teacher
                    teacher_logits = torch.matmul(teacher_img_features, student_txt_features.T)
                    student_logits = torch.matmul(student_img_features, student_txt_features.T)
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    student_log_probs = F.log_softmax(student_logits, dim=1)
                # KL divergence (teacher_probs is detached)
                soft_loss_img = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                loss = contrastive + SD_WEIGHT * soft_loss_img
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
            # Update EMA teacher (only image encoder)
            with torch.no_grad():
                for teacher_param, student_param in zip(teacher.image_encoder.parameters(), student.image_encoder.parameters()):
                    teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data
        # Save checkpoint for each epoch
        torch.save(student.state_dict(), os.path.join(checkpoint_dir, f"dynamicdistill_ft_epoch{epoch+1}.pth"))
    # Save final model
    torch.save(student.state_dict(), os.path.join(checkpoint_dir, "dynamicdistill_ft_final.pth"))
    return student

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Directory for saving checkpoints
    checkpoint_dir = "toy_exp_ckpts"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize models
    image_encoder = LightViT().to(device)
    text_encoder = LightTextTransformer().to(device)
    model = MultiModalContrastiveModel(image_encoder, text_encoder).to(device)
    
    print("=" * 50)
    print("Pre-training with contrastive learning...")
    print("=" * 50)
    
    # Pre-train the model
    train_losses, val_losses = pretrain_contrastive(model, train_loader, val_loader, epochs=10, checkpoint_dir=checkpoint_dir)
    
    # Save pre-trained model checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pretrained_multimodal.pth"))
    
    # Get templates for evaluation
    templates = train_mm.templates
    
    # Evaluate pre-trained model using mean embeddings
    print("\nPre-trained Model Accuracy (using mean embeddings):")
    print(f"Train (Image): {evaluate_with_mean_embeddings(model, train_loader, templates, 'image'):.2f}%")
    print(f"Val (Image): {evaluate_with_mean_embeddings(model, val_loader, templates, 'image'):.2f}%")
    print(f"Test (Image): {evaluate_with_mean_embeddings(model, test_loader, templates, 'image'):.2f}%")
    print(f"Train (Text): {evaluate_with_mean_embeddings(model, train_loader, templates, 'text'):.2f}%")
    print(f"Val (Text): {evaluate_with_mean_embeddings(model, val_loader, templates, 'text'):.2f}%")
    print(f"Test (Text): {evaluate_with_mean_embeddings(model, test_loader, templates, 'text'):.2f}%")
    
    # Store pre-trained accuracies for comparison
    pretrain_acc = {
        'train_img': evaluate_with_mean_embeddings(model, train_loader, templates, 'image'),
        'val_img': evaluate_with_mean_embeddings(model, val_loader, templates, 'image'),
        'test_img': evaluate_with_mean_embeddings(model, test_loader, templates, 'image'),
        'train_txt': evaluate_with_mean_embeddings(model, train_loader, templates, 'text'),
        'val_txt': evaluate_with_mean_embeddings(model, val_loader, templates, 'text'),
        'test_txt': evaluate_with_mean_embeddings(model, test_loader, templates, 'text'),
        # Evaluate on colored test set using classification head (to see spurious correlation learning)
        'test_colored_img': evaluate_with_mean_embeddings(model, test_colored_loader, templates, 'image')
    }
    

    
    print("\n" + "=" * 50)
    print("Fine-tuning with different strategies...")
    print("=" * 50)
    
    # Strategy 1: Direct fine-tuning
    print("\nStrategy 1: Direct Full Fine-tuning")
    model_direct = finetune_direct(model, train_colored_loader, val_colored_loader, epochs=10, checkpoint_dir=checkpoint_dir)
    torch.save(model_direct.state_dict(), os.path.join(checkpoint_dir, "finetuned_direct.pth"))
    direct_acc = {
        'train_img': evaluate_with_mean_embeddings(model_direct, train_loader, templates, 'image'),
        'val_img': evaluate_with_mean_embeddings(model_direct, val_loader, templates, 'image'),
        'test_img': evaluate_with_mean_embeddings(model_direct, test_loader, templates, 'image'),
        'train_txt': evaluate_with_mean_embeddings(model_direct, train_loader, templates, 'text'),
        'val_txt': evaluate_with_mean_embeddings(model_direct, val_loader, templates, 'text'),
        'test_txt': evaluate_with_mean_embeddings(model_direct, test_loader, templates, 'text'),
        # Evaluate on colored test set using classification head (to see spurious correlation learning)
        'test_colored_img': evaluate_with_mean_embeddings(model_direct, test_colored_loader, templates, 'image')
    }
    
    # Strategy 2: L2 regularization
    print("\nStrategy 2: L2 Regularization")
    model_l2 = finetune_l2_regularization(model, train_colored_loader, val_colored_loader, epochs=10, checkpoint_dir=checkpoint_dir)
    torch.save(model_l2.state_dict(), os.path.join(checkpoint_dir, "finetuned_l2reg.pth"))
    l2_acc = {
        'train_img': evaluate_with_mean_embeddings(model_l2, train_loader, templates, 'image'),
        'val_img': evaluate_with_mean_embeddings(model_l2, val_loader, templates, 'image'),
        'test_img': evaluate_with_mean_embeddings(model_l2, test_loader, templates, 'image'),
        'train_txt': evaluate_with_mean_embeddings(model_l2, train_loader, templates, 'text'),
        'val_txt': evaluate_with_mean_embeddings(model_l2, val_loader, templates, 'text'),
        'test_txt': evaluate_with_mean_embeddings(model_l2, test_loader, templates, 'text'),
        # Evaluate on colored test set using classification head (to see spurious correlation learning)
        'test_colored_img': evaluate_with_mean_embeddings(model_l2, test_colored_loader, templates, 'image')
    }
    
    # Strategy 3: Self-distillation
    print("\nStrategy 3: Self-distillation (Static Teacher)")
    model_distill = finetune_self_distillation(model, train_colored_loader, val_colored_loader, epochs=10, checkpoint_dir=checkpoint_dir)
    torch.save(model_distill.state_dict(), os.path.join(checkpoint_dir, "finetuned_selfdistill.pth"))
    distill_acc = {
        'train_img': evaluate_with_mean_embeddings(model_distill, train_loader, templates, 'image'),
        'val_img': evaluate_with_mean_embeddings(model_distill, val_loader, templates, 'image'),
        'test_img': evaluate_with_mean_embeddings(model_distill, test_loader, templates, 'image'),
        'train_txt': evaluate_with_mean_embeddings(model_distill, train_loader, templates, 'text'),
        'val_txt': evaluate_with_mean_embeddings(model_distill, val_loader, templates, 'text'),
        'test_txt': evaluate_with_mean_embeddings(model_distill, test_loader, templates, 'text'),
        # Evaluate on colored test set using classification head (to see spurious correlation learning)
        'test_colored_img': evaluate_with_mean_embeddings(model_distill, test_colored_loader, templates, 'image')
    }
    
    # Strategy 4: Dynamic self-distillation
    print("\nStrategy 4: Dynamic Self-distillation (EMA Teacher)")
    model_dynamic = finetune_dynamic_self_distillation(model, train_colored_loader, val_colored_loader, epochs=10, checkpoint_dir=checkpoint_dir)
    torch.save(model_dynamic.state_dict(), os.path.join(checkpoint_dir, "finetuned_dynamicdistill.pth"))
    dynamic_acc = {
        'train_img': evaluate_with_mean_embeddings(model_dynamic, train_loader, templates, 'image'),
        'val_img': evaluate_with_mean_embeddings(model_dynamic, val_loader, templates, 'image'),
        'test_img': evaluate_with_mean_embeddings(model_dynamic, test_loader, templates, 'image'),
        'train_txt': evaluate_with_mean_embeddings(model_dynamic, train_loader, templates, 'text'),
        'val_txt': evaluate_with_mean_embeddings(model_dynamic, val_loader, templates, 'text'),
        'test_txt': evaluate_with_mean_embeddings(model_dynamic, test_loader, templates, 'text'),
        # Evaluate on colored test set using classification head (to see spurious correlation learning)
        'test_colored_img': evaluate_with_mean_embeddings(model_dynamic, test_colored_loader, templates, 'image')
    }
    
    # ==================== Results Summary ====================
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    # Create comparison table
    methods = ['Pre-trained', 'Direct FT', 'L2 Reg', 'Static Distill', 'Dynamic Distill']
    results = [pretrain_acc, direct_acc, l2_acc, distill_acc, dynamic_acc]
    
    print("\nImage Modality Accuracy (%) - Original MNIST:")
    print(f"{'Method':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 60)
    for method, res in zip(methods, results):
        print(f"{method:<20} {res['train_img']:<10.2f} {res['val_img']:<10.2f} {res['test_img']:<10.2f}")
    
    print("\nImage Modality Accuracy (%) - Colored MNIST Test (with spurious correlation):")
    print(f"{'Method':<20} {'Test Colored':<15}")
    print("-" * 40)
    for method, res in zip(methods, results):
        print(f"{method:<20} {res['test_colored_img']:<15.2f}")
    
    print("\nText Modality Accuracy (%):")
    print(f"{'Method':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 60)
    for method, res in zip(methods, results):
        print(f"{method:<20} {res['train_txt']:<10.2f} {res['val_txt']:<10.2f} {res['test_txt']:<10.2f}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Image modality plot - Original MNIST
    x = np.arange(len(methods))
    width = 0.25
    
    train_scores_img = [res['train_img'] for res in results]
    val_scores_img = [res['val_img'] for res in results]
    test_scores_img = [res['test_img'] for res in results]
    
    ax1.bar(x - width, train_scores_img, width, label='Train')
    ax1.bar(x, val_scores_img, width, label='Val')
    ax1.bar(x + width, test_scores_img, width, label='Test')
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Image Modality - Original MNIST')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Colored MNIST test comparison
    test_colored_scores = [res['test_colored_img'] for res in results]
    
    # Use offset bars to show both side by side
    bar_width = 0.35
    ax2.bar(x - bar_width/2, test_colored_scores, bar_width, label='Test Colored', color='orange')
    ax2.bar(x + bar_width/2, test_scores_img, bar_width, label='Test Original', color='blue')
    ax2.set_xlabel('Methods')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Text modality plot
    train_scores_txt = [res['train_txt'] for res in results]
    val_scores_txt = [res['val_txt'] for res in results]
    test_scores_txt = [res['test_txt'] for res in results]
    
    ax3.bar(x - width, train_scores_txt, width, label='Train')
    ax3.bar(x, val_scores_txt, width, label='Val')
    ax3.bar(x + width, test_scores_txt, width, label='Test')
    ax3.set_xlabel('Methods')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Text Modality Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('finetuning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)
    
    # Calculate average improvement over pre-trained model
    for i, (method, res) in enumerate(zip(methods[1:], results[1:]), 1):
        img_improvement = res['test_img'] - pretrain_acc['test_img']
        txt_improvement = res['test_txt'] - pretrain_acc['test_txt']
        colored_improvement = res['test_colored_img'] - pretrain_acc['test_colored_img']
        
        print(f"\n{method}:")
        print(f"  Original test improvement: {img_improvement:+.2f}%")
        print(f"  Colored test improvement: {colored_improvement:+.2f}%")
        print(f"  Text improvement: {txt_improvement:+.2f}%")
        
        # Check for spurious correlation reliance
        spurious_gap = res['test_colored_img'] - res['test_img']
        print(f"  Spurious correlation gap: {spurious_gap:+.2f}%")
        if spurious_gap > 5:
            print(f"    -> Heavily relies on spurious correlation")
        elif spurious_gap > 0:
            print(f"    -> Moderately relies on spurious correlation")
        else:
            print(f"    -> Robust to spurious correlation")
        
        # Check for overfitting
        img_overfit = res['train_img'] - res['test_img']
        txt_overfit = res['train_txt'] - res['test_txt']
        print(f"  Image overfitting gap: {img_overfit:.2f}%")
        print(f"  Text overfitting gap: {txt_overfit:.2f}%")