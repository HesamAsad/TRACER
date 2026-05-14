import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional, List
import itertools
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ================== DATA GENERATION ==================

class ColoredMNISTPlus(Dataset):
    """Extended Colored MNIST with controllable subspace coverage and spurious correlations."""
    
    # Dictionary for encoding values
    DICT = {
        0: -4.5, 1: -3.5, 2: -2.5, 3: -1.5, 4: -0.5,
        5: 0.5, 6: 1.5, 7: 2.5, 8: 3.5, 9: 4.5,
        "dark_blue": -2.5, "medium_blue": -1.5, "light_blue": -0.5,
        "light_red": 0.5, "medium_red": 1.5, "dark_red": 2.5,
        "solid": -1.0, "striped": 1.0
    }
    
    # Color definitions (RGB)
    COLORS = {
        "dark_blue": (0, 0, 139),
        "medium_blue": (0, 0, 205),
        "light_blue": (135, 206, 250),
        "light_red": (255, 182, 193),
        "medium_red": (220, 20, 60),
        "dark_red": (139, 0, 0)
    }
    
    def __init__(self, 
                 split='train',
                 rho=0.995,  # Spurious correlation strength
                 K_rot=4,    # Number of rotation angles
                 K_dig=5,    # Number of distinct digits per class
                 K_stroke=3, # Number of stroke widths
                 include_stripes=True,
                 pi_core=0.8,
                 pi_spu_color=0.8,
                 pi_spu_pattern=0.8,
                 n_samples=10000,
                 orthogonal_test=False):
        
        self.split = split
        self.rho = rho
        self.K_rot = K_rot
        self.K_dig = K_dig
        self.K_stroke = K_stroke
        self.include_stripes = include_stripes
        self.pi_core = pi_core
        self.pi_spu_color = pi_spu_color
        self.pi_spu_pattern = pi_spu_pattern
        self.n_samples = n_samples
        self.orthogonal_test = orthogonal_test
        
        # Load MNIST
        from torchvision import datasets, transforms
        mnist = datasets.MNIST('./data', train=(split in ['train', 'finetune']), 
                              download=True)
        self.mnist_data = mnist.data.numpy()
        self.mnist_labels = mnist.targets.numpy()
        
        # Generate dataset
        self.data, self.labels, self.captions, self.metadata = self._generate_data()
    
    def _generate_data(self):
        """Generate colored MNIST++ data with controlled properties."""
        data = []
        labels = []
        captions = []
        metadata = []
        
        # Define rotation angles based on K_rot
        if self.orthogonal_test:
            # Use angles NOT in training set
            base_angles = np.linspace(0, 360, 32, endpoint=False)
            train_angles = np.linspace(0, 360, self.K_rot, endpoint=False)
            angles = [a for a in base_angles if not any(abs(a - ta) < 5 for ta in train_angles)]
        else:
            angles = np.linspace(0, 360, self.K_rot, endpoint=False)
        
        # Define stroke widths - more conservative range to avoid digit disappearance
        if self.K_stroke == 1:
            stroke_widths = [1.0]
        else:
            stroke_widths = np.linspace(0.7, 1.8, self.K_stroke)  # Reduced range
        
        # Select digits for each class based on K_dig
        class_A_digits = sorted(np.random.choice(range(5), min(self.K_dig, 5), replace=False))
        class_B_digits = sorted(np.random.choice(range(5, 10), min(self.K_dig, 5), replace=False))
        
        blue_colors = ["dark_blue", "medium_blue", "light_blue"]
        red_colors = ["light_red", "medium_red", "dark_red"]
        
        for _ in range(self.n_samples):
            # Sample class
            class_label = np.random.randint(2)  # 0 or 1
            
            # Sample digit based on class
            if class_label == 0:
                digit = np.random.choice(class_A_digits)
            else:
                digit = np.random.choice(class_B_digits)
            
            # Get MNIST image for this digit
            digit_indices = np.where(self.mnist_labels == digit)[0]
            idx = np.random.choice(digit_indices)
            img = self.mnist_data[idx]
            
            # Apply transformations
            angle = np.random.choice(angles)
            stroke_width = np.random.choice(stroke_widths)
            
            # Apply rotation
            img = self._rotate_image(img, angle)
            
            # Apply stroke width modification
            img = self._modify_stroke_width(img, stroke_width)
            
            # Determine color based on spurious correlation
            if self.split in ['train', 'finetune', 'id_test']:
                # Apply spurious correlation
                if np.random.rand() < self.rho:
                    # Correlated
                    color = np.random.choice(blue_colors if class_label == 0 else red_colors)
                    pattern = "solid" if class_label == 0 else "striped"
                else:
                    # Anti-correlated
                    color = np.random.choice(red_colors if class_label == 0 else blue_colors)
                    pattern = "striped" if class_label == 0 else "solid"
            elif self.split == 'ood_color':
                # Random color, correlated pattern
                color = np.random.choice(blue_colors + red_colors)
                if np.random.rand() < self.rho:
                    pattern = "solid" if class_label == 0 else "striped"
                else:
                    pattern = "striped" if class_label == 0 else "solid"
            elif self.split == 'ood_pattern':
                # Correlated color, random pattern
                if np.random.rand() < self.rho:
                    color = np.random.choice(blue_colors if class_label == 0 else red_colors)
                else:
                    color = np.random.choice(red_colors if class_label == 0 else blue_colors)
                pattern = np.random.choice(["solid", "striped"])
            else:  # ood_both
                # Both random
                color = np.random.choice(blue_colors + red_colors)
                pattern = np.random.choice(["solid", "striped"])
            
            if not self.include_stripes:
                pattern = "solid"
            
            # Apply color and pattern
            colored_img = self._apply_color_and_pattern(img, color, pattern)
            
            # Generate caption vector
            caption = self._generate_caption(digit, color, pattern)
            
            data.append(colored_img)
            labels.append(class_label)
            captions.append(caption)
            metadata.append({
                'digit': digit,
                'color': color,
                'pattern': pattern,
                'angle': angle,
                'stroke_width': stroke_width
            })
        
        return np.array(data), np.array(labels), np.array(captions), metadata
    
    def _rotate_image(self, img, angle):
        """Rotate image by given angle."""
        from scipy.ndimage import rotate
        return rotate(img, angle, reshape=False, order=1)
    
    def _modify_stroke_width(self, img, width):
        """Modify stroke width of digit."""
        from scipy.ndimage import binary_dilation, binary_erosion
        
        img_binary = img > 128
        if width > 1:
            # Dilate for thicker strokes
            iterations = max(1, int((width - 1) * 1.5))  # Reduced multiplier
            img_binary = binary_dilation(img_binary, iterations=iterations)
        elif width < 1:
            # Erode for thinner strokes - be more conservative
            iterations = max(1, min(2, int((1 - width) * 1.5)))  # Limited erosion
            img_binary = binary_erosion(img_binary, iterations=iterations)
        
        # Ensure we don't completely lose the digit
        if np.sum(img_binary) < 10:  # If too few pixels remain, use original
            img_binary = img > 128
            
        return img_binary.astype(np.float32) * 255
    
    def _apply_color_and_pattern(self, img, color, pattern):
        """Apply color and pattern to grayscale image."""
        # Normalize image
        img = img / 255.0
        
        # Create RGB image
        h, w = img.shape
        rgb_img = np.zeros((h, w, 3))
        
        # Get color RGB values
        color_rgb = np.array(self.COLORS[color]) / 255.0
        
        # Create background with lighter intensity for better contrast
        bg_intensity = 0.15  # Reduced from 0.3 for better contrast
        if pattern == "striped":
            # Create vertical stripes
            stripe_width = 4
            for i in range(0, w, stripe_width * 2):
                rgb_img[:, i:i+stripe_width] = color_rgb * bg_intensity
        else:
            # Solid background
            rgb_img[:, :] = color_rgb * bg_intensity
        
        # Create digit mask
        digit_mask = img > 0.1  # Lower threshold to capture more of the digit
        
        # Apply digit with better contrast
        # Use white/light color for digits to ensure visibility
        digit_color = np.array([0.9, 0.9, 0.9])  # Light gray/white for digits
        
        for c in range(3):
            # Blend digit color with slight tint from background color
            final_digit_color = 0.8 * digit_color[c] + 0.2 * color_rgb[c]
            rgb_img[:, :, c] = np.where(digit_mask, 
                                        img * final_digit_color + rgb_img[:, :, c] * (1 - img),
                                        rgb_img[:, :, c])
        
        return rgb_img
    
    def _generate_caption(self, digit, color, pattern):
        """Generate caption vector with controlled richness."""
        # Determine values based on richness parameters
        if np.random.rand() < self.pi_core:
            a = self.DICT[digit]
        else:
            # Use group mean
            a = -2.5 if digit <= 4 else 2.5
        
        if np.random.rand() < self.pi_spu_color:
            b = self.DICT[color]
        else:
            # Use group mean
            if "blue" in color:
                b = -1.5
            else:
                b = 1.5
        
        if np.random.rand() < self.pi_spu_pattern:
            p = self.DICT[pattern]
        else:
            # Use mean (0 for pattern)
            p = 0.0
        
        # Create vector
        v = np.zeros(200)
        v[0] = a
        v[1] = b
        v[2] = p
        
        # Add noise
        noise = np.random.normal(0, 1/np.sqrt(2000), size=200)
        
        return v + noise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to torch tensors
        img = torch.FloatTensor(self.data[idx]).permute(2, 0, 1)  # CHW format
        caption = torch.FloatTensor(self.captions[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        return img, caption, label

# ================== MODELS ==================

class LeNetEncoder(nn.Module):
    """LeNet-style CNN encoder."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=1)

class TextEncoder(nn.Module):
    """Linear text encoder."""
    
    def __init__(self, input_dim=200, output_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return F.normalize(self.linear(x), dim=1)

class CLIPModel(nn.Module):
    """CLIP model combining vision and text encoders."""
    
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones(1) * np.log(1/0.07))
        
    def forward(self, images, texts):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        return image_features, text_features

# ================== TRAINING METHODS ==================

class FineTuningMethod:
    """Base class for fine-tuning methods."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.initial_vision_state = {k: v.clone() for k, v in 
                                    model.vision_encoder.state_dict().items()}
        
    def compute_loss(self, images, captions, labels):
        raise NotImplementedError
    
    def update_step(self, loss, optimizer):
        """Standard update step."""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def clip_classification_loss(image_features, text_features, labels, temperature):
    """
    Computes a classification-oriented contrastive loss.
    It creates on-the-fly text prototypes for each class present in the batch.
    """
    labels = labels.squeeze()
    unique_labels = torch.unique(labels)

    # If batch contains only one class, loss is 0 as no contrast is possible
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    # Create text prototypes by averaging features of each class
    prototype_text_features = torch.stack([
        text_features[labels == l].mean(dim=0) for l in unique_labels
    ])
    prototype_text_features = F.normalize(prototype_text_features, dim=1)

    # Compute logits: similarity of each image feature to each class prototype
    logits = image_features @ prototype_text_features.t() * torch.exp(temperature)

    # Remap original labels to match the order of prototypes for cross_entropy
    remapped_labels = torch.zeros_like(labels)
    for i, l in enumerate(unique_labels):
        remapped_labels[labels == l] = i
        
    return F.cross_entropy(logits, remapped_labels)

class DirectFineTuning(FineTuningMethod):
    """Direct fine-tuning (FLYP-style)."""
    
    def compute_loss(self, images, captions, labels):
        image_features, text_features = self.model(images, captions)
        return clip_classification_loss(image_features, text_features, labels, self.model.temperature)

class L2SPFineTuning(FineTuningMethod):
    """L2-SP regularized fine-tuning."""
    
    def __init__(self, model, lambda_reg=0.01, device='cuda'):
        super().__init__(model, device)
        self.lambda_reg = lambda_reg
    
    def compute_loss(self, images, captions, labels):
        # Standard CLIP loss
        image_features, text_features = self.model(images, captions)
        clip_loss = clip_classification_loss(image_features, text_features, labels, self.model.temperature)
        
        # L2-SP regularization
        l2_loss = 0
        for name, param in self.model.vision_encoder.named_parameters():
            l2_loss += torch.sum((param - self.initial_vision_state[name]) ** 2)
        
        return clip_loss + self.lambda_reg * l2_loss / 2

class StaticSelfDistillation(FineTuningMethod):
    """Static self-distillation."""
    
    def __init__(self, model, lambda_reg=0.01, device='cuda'):
        super().__init__(model, device)
        self.lambda_reg = lambda_reg
        
        # Create frozen teacher
        self.teacher = LeNetEncoder().to(device)
        self.teacher.load_state_dict(self.initial_vision_state)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def compute_loss(self, images, captions, labels):
        # Standard CLIP loss
        image_features, text_features = self.model(images, captions)
        clip_loss = clip_classification_loss(image_features, text_features, labels, self.model.temperature)
        
        # Self-distillation loss
        with torch.no_grad():
            teacher_features = self.teacher(images)
        
        distill_loss = F.mse_loss(image_features, teacher_features)
        
        return clip_loss + self.lambda_reg * distill_loss

class SDBMAFineTuning(FineTuningMethod):
    """Self-distillation with Beta moving average."""
    
    def __init__(self, model, beta=0.9, alpha=0.5, device='cuda'):
        super().__init__(model, device)
        self.beta = beta
        self.alpha = alpha
        
        # Create teacher as copy of student
        self.teacher = LeNetEncoder().to(device)
        self.teacher.load_state_dict(self.initial_vision_state)
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def compute_loss(self, images, captions, labels):
        # Standard CLIP loss
        image_features, text_features = self.model(images, captions)
        clip_loss = clip_classification_loss(image_features, text_features, labels, self.model.temperature)
        
        # Self-distillation loss
        with torch.no_grad():
            teacher_features = self.teacher(images)
        
        distill_loss = F.mse_loss(image_features, teacher_features)
        
        return (1 - self.alpha) * clip_loss + self.alpha * distill_loss
    
    def update_step(self, loss, optimizer):
        """Update including teacher EMA update."""
        # Standard gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher with EMA
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), 
                                                    self.model.vision_encoder.parameters()):
                teacher_param.data = self.beta * teacher_param.data + (1 - self.beta) * student_param.data

# ================== EVALUATION AND METRICS ==================

def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate model classification accuracy using more robust prototype captions."""
    model.eval()

    # Create prototype captions for each digit, then average their features for class prototypes
    class_A_digits = range(5)
    class_B_digits = range(5, 10)
    
    # Generate a caption vector for each digit, with spurious features zeroed out
    proto_captions_A = torch.zeros(len(class_A_digits), 200).to(device)
    for i, digit in enumerate(class_A_digits):
        proto_captions_A[i, 0] = ColoredMNISTPlus.DICT[digit]

    proto_captions_B = torch.zeros(len(class_B_digits), 200).to(device)
    for i, digit in enumerate(class_B_digits):
        proto_captions_B[i, 0] = ColoredMNISTPlus.DICT[digit]
        
    with torch.no_grad():
        # Get text features for each digit caption
        features_A = model.text_encoder(proto_captions_A)
        features_B = model.text_encoder(proto_captions_B)
        
        # Average the features to get a single, robust prototype per class
        proto_A = F.normalize(features_A.mean(dim=0, keepdim=True))
        proto_B = F.normalize(features_B.mean(dim=0, keepdim=True))
        
        # Combine into one tensor for easy matrix multiplication
        prototype_text_features = torch.cat([proto_A, proto_B], dim=0)

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, _, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).squeeze()
            
            image_features = model.vision_encoder(images)
            
            # Classify based on similarity to the robust class prototypes
            logits = image_features @ prototype_text_features.t()
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += len(images)
    
    return correct / total if total > 0 else 0

def compute_projection_metrics(model, initial_model, train_loader, device='cuda'):
    """Compute projection-based metrics."""
    
    # Collect training data
    X_I = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in train_loader:
            images = images.to(device)
            features = model.vision_encoder(images)
            X_I.append(features.cpu().numpy())

    X_I = np.concatenate(X_I, axis=0)  # Shape: (n, d)
    X_I = X_I.T  # Shape: (d, n)

    # Compute projector P_I = X_I (X_I^T X_I)^+ X_I^T
    # For numerical stability, use SVD
    U, S, Vt = np.linalg.svd(X_I, full_matrices=False)
    threshold = 1e-10
    S_inv = np.where(S > threshold, 1/S, 0)
    X_I_pinv = Vt.T @ np.diag(S_inv) @ U.T  # (n, d)
    P_I = X_I @ X_I_pinv  # (d, d)
    
    # Compute weight difference for the vision projection layer mapping feature_dim -> embed_dim
    # Identify a 2D weight whose second dim matches feature_dim (d)
    d = P_I.shape[0]
    delta_W_proj = None
    for name, param in model.vision_encoder.named_parameters():
        if name in initial_model:
            w = param.data.cpu().numpy()
            if w.ndim == 2 and w.shape[1] == d:
                delta_W_proj = w - initial_model[name].cpu().numpy()
                break

    if delta_W_proj is None:
        # Fallback: try to find any 2D weight and align last dim if possible
        for name, param in model.vision_encoder.named_parameters():
            if name in initial_model:
                w = param.data.cpu().numpy()
                if w.ndim == 2:
                    if w.shape[1] >= d:
                        # Use last d columns as feature-aligned block
                        delta_W_proj = (w - initial_model[name].cpu().numpy())[:, -d:]
                        break
        if delta_W_proj is None:
            return {
                'norm_in_subspace': 0.0,
                'norm_orthogonal': 0.0,
                'fraction_orthogonal': 0.0
            }

    # Project the weight difference onto data subspace: ΔW P_I and ΔW (I - P_I)
    delta_W_P = delta_W_proj @ P_I  # (embed_dim, d)
    delta_W_I_minus_P = delta_W_proj @ (np.eye(d) - P_I)

    # Frobenius norms
    norm_P = np.linalg.norm(delta_W_P)
    norm_I_minus_P = np.linalg.norm(delta_W_I_minus_P)
    norm_total = np.linalg.norm(delta_W_proj)

    return {
        'norm_in_subspace': float(norm_P),
        'norm_orthogonal': float(norm_I_minus_P),
        'fraction_orthogonal': float(norm_I_minus_P / norm_total) if norm_total > 0 else 0.0
    }

# ================== MAIN EXPERIMENT RUNNER ==================

class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
    def pretrain_model(self, n_epochs=100):
        """Pretrain CLIP on uncorrelated MNIST."""
        print("Pretraining CLIP model...")
        
        # Create pretraining dataset (no correlations)
        pretrain_data = ColoredMNISTPlus(
            split='train',
            rho=0.5,  # No correlation
            K_rot=16,
            K_dig=5,
            K_stroke=3,
            include_stripes=True,
            pi_core=1.0,
            pi_spu_color=1.0,
            pi_spu_pattern=1.0,
            n_samples=20000
        )
        
        pretrain_loader = DataLoader(pretrain_data, batch_size=128, shuffle=True)
        
        # Create model
        vision_encoder = LeNetEncoder()
        text_encoder = TextEncoder()
        model = CLIPModel(vision_encoder, text_encoder).to(self.device)
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for images, captions, labels in pretrain_loader:
                images = images.to(self.device)
                captions = captions.to(self.device)
                labels = labels.to(self.device)
                
                image_features, text_features = model(images, captions)
                loss = clip_classification_loss(image_features, text_features, labels, model.temperature)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Pretrain Epoch {epoch}: Loss = {total_loss/len(pretrain_loader):.4f}")
        
        return model
    
    def run_finetuning(self, pretrained_model, method_class, method_kwargs, 
                      dataset_kwargs, n_epochs=100):
        """Run fine-tuning with specified method."""
        
        # Create fine-tuning dataset
        finetune_data = ColoredMNISTPlus(split='finetune', **dataset_kwargs)
        finetune_loader = DataLoader(finetune_data, batch_size=128, shuffle=True)
        
        # Clone model for fine-tuning
        vision_encoder = LeNetEncoder()
        vision_encoder.load_state_dict(pretrained_model.vision_encoder.state_dict())
        text_encoder = TextEncoder()
        text_encoder.load_state_dict(pretrained_model.text_encoder.state_dict())
        model = CLIPModel(vision_encoder, text_encoder).to(self.device)
        
        # Create fine-tuning method
        method = method_class(model, **method_kwargs, device=self.device)
        
        # Optimizer (only vision encoder for fine-tuning)
        optimizer = optim.SGD(model.vision_encoder.parameters(), 
                            lr=0.01, momentum=0.9, weight_decay=1e-3)
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for images, captions, labels in finetune_loader:
                images = images.to(self.device)
                captions = captions.to(self.device)
                labels = labels.to(self.device)
                
                loss = method.compute_loss(images, captions, labels)
                method.update_step(loss, optimizer)
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Finetune Epoch {epoch}: Loss = {total_loss/len(finetune_loader):.4f}")
        
        return model, method
    
    def evaluate_all_splits(self, model, dataset_kwargs):
        """Evaluate model on all test splits."""
        results = {}
        
        splits = ['id_test', 'ood_color', 'ood_pattern', 'ood_both']
        
        for split in splits:
            test_kwargs = dataset_kwargs.copy()
            test_kwargs['n_samples'] = 2000
            test_data = ColoredMNISTPlus(split=split, **test_kwargs)
            test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
            acc = evaluate_model(model, test_loader, self.device)
            results[split] = acc
        
        # Orthogonal subspace test
        os_kwargs = dataset_kwargs.copy()
        os_kwargs['n_samples'] = 2000
        os_data = ColoredMNISTPlus(split='id_test', orthogonal_test=True, **os_kwargs)
        os_loader = DataLoader(os_data, batch_size=128, shuffle=False)
        results['orthogonal_subspace'] = evaluate_model(model, os_loader, self.device)
        
        return results
    
    def experiment_1_projector_test(self):
        """E1: Where does forgetting happen?"""
        print("\n" + "="*50)
        print("EXPERIMENT 1: Projector Test")
        print("="*50)
        
        # Pretrain model
        pretrained_model = self.pretrain_model(n_epochs=60)
        
        # Fixed caption richness
        base_kwargs = {
            'rho': 0.995,
            'pi_core': 0.8,
            'pi_spu_color': 0.8,
            'pi_spu_pattern': 0.8,
            'K_stroke': 3,
            'include_stripes': True,
            'n_samples': 5000
        }
        
        # Sweep subspace coverage
        K_rot_values = [1, 2, 4, 16]
        K_dig_values = [1, 2, 5]
        
        results = []
        
        for K_rot in K_rot_values:
            for K_dig in K_dig_values:
                dataset_kwargs = base_kwargs.copy()
                dataset_kwargs['K_rot'] = K_rot
                dataset_kwargs['K_dig'] = K_dig
                
                print(f"\nTesting K_rot={K_rot}, K_dig={K_dig}")
                
                # Test each method
                methods = [
                    ('Direct FT', DirectFineTuning, {}),
                    ('L2-SP (λ=0.001)', L2SPFineTuning, {'lambda_reg': 0.001}),
                    ('SD-Static (λ=0.001)', StaticSelfDistillation, {'lambda_reg': 0.001}),
                    ('SD-BMA (β=0.9, α=0.1)', SDBMAFineTuning, {'beta': 0.9, 'alpha': 0.1}),
                    ('SD-BMA (β=0.99, α=0.1)', SDBMAFineTuning, {'beta': 0.99, 'alpha': 0.1}),
                    ('SD-BMA (β=0.999, α=0.1)', SDBMAFineTuning, {'beta': 0.999, 'alpha': 0.1}),
                    ('SD-BMA (β=0.9, α=0.9)', SDBMAFineTuning, {'beta': 0.9, 'alpha': 0.9}),
                    ('SD-BMA (β=0.99, α=0.9)', SDBMAFineTuning, {'beta': 0.99, 'alpha': 0.9}),
                    ('SD-BMA (β=0.999, α=0.9)', SDBMAFineTuning, {'beta': 0.999, 'alpha': 0.9}),
                    ('SD-BMA (β=0.9, α=0.99)', SDBMAFineTuning, {'beta': 0.9, 'alpha': 0.99}),
                    ('SD-BMA (β=0.99, α=0.99)', SDBMAFineTuning, {'beta': 0.99, 'alpha': 0.99}),
                    ('SD-BMA (β=0.999, α=0.99)', SDBMAFineTuning, {'beta': 0.999, 'alpha': 0.99})
                ]
                
                for method_name, method_class, method_kwargs in methods:
                    print(f"  Running {method_name}...")
                    
                    model, method = self.run_finetuning(
                        pretrained_model, method_class, method_kwargs, 
                        dataset_kwargs, n_epochs=50
                    )
                    
                    # Evaluate
                    eval_results = self.evaluate_all_splits(model, dataset_kwargs)
                    
                    # Compute projection metrics
                    finetune_data = ColoredMNISTPlus(split='finetune', **dataset_kwargs)
                    finetune_loader = DataLoader(finetune_data, batch_size=128)
                    proj_metrics = compute_projection_metrics(
                        model, method.initial_vision_state, finetune_loader, self.device
                    )
                    
                    results.append({
                        'K_rot': K_rot,
                        'K_dig': K_dig,
                        'method': method_name,
                        **eval_results,
                        **proj_metrics
                    })
        
        self.results['experiment_1'] = pd.DataFrame(results)
        return self.results['experiment_1']
    
    def experiment_2_caption_richness(self):
        """E2: Caption richness = target matrix geometry."""
        print("\n" + "="*50)
        print("EXPERIMENT 2: Caption Richness")
        print("="*50)
        
        # Pretrain model
        pretrained_model = self.pretrain_model(n_epochs=40) # TODO: maybe increase epochs?
        
        # Fixed coverage
        base_kwargs = {
            'rho': 0.995,
            'K_rot': 4,
            'K_dig': 5,
            'K_stroke': 3,
            'include_stripes': True,
            'n_samples': 5000
        }
        
        # Sweep caption richness
        pi_core_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        pi_spu_values = [0.0, 0.5, 1.0]
        
        results = []
        
        for pi_core in pi_core_values:
            for pi_spu in pi_spu_values:
                dataset_kwargs = base_kwargs.copy()
                dataset_kwargs['pi_core'] = pi_core
                dataset_kwargs['pi_spu_color'] = pi_spu
                dataset_kwargs['pi_spu_pattern'] = pi_spu
                
                print(f"\nTesting pi_core={pi_core}, pi_spu={pi_spu}")
                
                # Test key methods
                methods = [
                    ('Direct FT', DirectFineTuning, {}),
                    ('L2-SP', L2SPFineTuning, {'lambda_reg': 0.01}),
                    ('SD-BMA', SDBMAFineTuning, {'beta': 0.9, 'alpha': 0.5})
                ]
                
                for method_name, method_class, method_kwargs in methods:
                    print(f"  Running {method_name}...")
                    
                    model, _ = self.run_finetuning(
                        pretrained_model, method_class, method_kwargs,
                        dataset_kwargs, n_epochs=50
                    )
                    
                    # Evaluate
                    eval_results = self.evaluate_all_splits(model, dataset_kwargs)
                    
                    results.append({
                        'pi_core': pi_core,
                        'pi_spu': pi_spu,
                        'method': method_name,
                        **eval_results
                    })
        
        self.results['experiment_2'] = pd.DataFrame(results)
        return self.results['experiment_2']
    
    def plot_results(self):
        """Generate all plots for the paper."""
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: OOD-Both vs K_rot (E1)
        if 'experiment_1' in self.results:
            df = self.results['experiment_1']
            
            ax1 = plt.subplot(2, 3, 1)
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                # Average over K_dig for each K_rot
                avg_df = method_df.groupby('K_rot')['ood_both'].mean()
                ax1.plot(avg_df.index, avg_df.values, marker='o', label=method, linewidth=2)
            
            ax1.set_xlabel('K_rot (Number of rotations)', fontsize=12)
            ax1.set_ylabel('OOD-Both Accuracy', fontsize=12)
            ax1.set_title('(a) Robustness vs Subspace Coverage', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: OS retention vs K_rot
            ax2 = plt.subplot(2, 3, 2)
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                avg_df = method_df.groupby('K_rot')['orthogonal_subspace'].mean()
                ax2.plot(avg_df.index, avg_df.values, marker='s', label=method, linewidth=2)
            
            ax2.set_xlabel('K_rot (Number of rotations)', fontsize=12)
            ax2.set_ylabel('Orthogonal Subspace Retention', fontsize=12)
            ax2.set_title('(b) Preservation Outside Subspace', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Projection metrics bar plot
            ax3 = plt.subplot(2, 3, 3)
            # Average over all configurations
            avg_proj = df.groupby('method')[['norm_in_subspace', 'norm_orthogonal']].mean()
            
            x = np.arange(len(avg_proj.index))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, avg_proj['norm_in_subspace'], width, 
                          label='||ΔW P_I||', color='steelblue')
            bars2 = ax3.bar(x + width/2, avg_proj['norm_orthogonal'], width,
                          label='||ΔW (I-P_I)||', color='coral')
            
            ax3.set_xlabel('Method', fontsize=12)
            ax3.set_ylabel('Norm', fontsize=12)
            ax3.set_title('(c) Projection Decomposition', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(avg_proj.index, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4-6: Heatmaps for E2
        if 'experiment_2' in self.results:
            df = self.results['experiment_2']
            
            methods_to_plot = ['Direct FT', 'L2-SP', 'SD-BMA']
            
            for i, method in enumerate(methods_to_plot):
                ax = plt.subplot(2, 3, 4 + i)
                
                method_df = df[df['method'] == method]
                pivot = method_df.pivot(index='pi_spu', columns='pi_core', values='ood_both')
                
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                          vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'OOD-Both Acc'})
                
                ax.set_xlabel('π_core', fontsize=12)
                ax.set_ylabel('π_spu', fontsize=12)
                ax.set_title(f'(d{i+1}) {method}: Caption Richness Effect', 
                           fontsize=14, fontweight='bold')
        
        plt.suptitle('Semi-Synthetic CLIP Fine-Tuning: Validating Theoretical Predictions',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def generate_summary_table(self):
        """Generate summary table of results."""
        
        if 'experiment_1' not in self.results:
            return None
        
        df = self.results['experiment_1']
        
        # Aggregate metrics
        summary = df.groupby('method').agg({
            'id_test': 'mean',
            'ood_color': 'mean',
            'ood_pattern': 'mean',
            'ood_both': 'mean',
            'orthogonal_subspace': 'mean',
            'fraction_orthogonal': 'mean'
        }).round(3)
        
        # Add standard deviations (population std, ddof=0 to avoid NaN when single sample)
        std = df.groupby('method').agg(
            ood_both_std=('ood_both', lambda x: float(np.std(x, ddof=0))),
            os_std=('orthogonal_subspace', lambda x: float(np.std(x, ddof=0)))
        )
        std = std.round(3)
        summary = pd.concat([summary, std], axis=1).fillna(0.0)
        
        return summary

# ================== MAIN EXECUTION ==================

def main():
    """Run all experiments and generate results."""
    
    print("="*60)
    print("SEMI-SYNTHETIC CLIP FINE-TUNING EXPERIMENTS")
    print("Validating Target-Matrix View and Subspace Projector Theory")
    print("="*60)
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Run experiments
    print("\nRunning experiments...")
    
    # Experiment 1: Projector test
    exp1_results = runner.experiment_1_projector_test()
    print("\nExperiment 1 Results:")
    print(exp1_results.head(10))
    
    # Experiment 2: Caption richness
    exp2_results = runner.experiment_2_caption_richness()
    print("\nExperiment 2 Results:")
    print(exp2_results.head(10))
    
    # Generate plots
    print("\nGenerating plots...")
    fig = runner.plot_results()
    plt.savefig('clip_finetuning_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'clip_finetuning_results.png'")
    
    # Generate summary table
    print("\nSummary Table:")
    summary = runner.generate_summary_table()
    if summary is not None:
        print(summary)
        summary.to_csv('summary_results.csv')
        print("Summary saved to 'summary_results.csv'")
    
    # Save detailed results
    for name, df in runner.results.items():
        filename = f'{name}_detailed.csv'
        df.to_csv(filename, index=False)
        print(f"Detailed results saved to '{filename}'")
    
    plt.show()
    
    return runner

if __name__ == "__main__":
    runner = main()