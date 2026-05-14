import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F

import timm
from transformers import AutoModel, AutoTokenizer, AutoConfig

import numpy as np
from tqdm import tqdm
import random
import copy
import pandas as pd
import argparse

# 1. Configuration
class Config:
    # Data
    IMG_SIZE = (224, 224) # ViT standard size
    CAPTIONS = [
        "the digit {}", "number {}", "handwritten {}", 
        "a {} digit", "the number {} written by hand"
    ]
    
    # Model
    EMBED_DIM = 8
    TEXT_MODEL_NAME = 'prajjwal1/bert-tiny'
    VISION_MODEL_NAME = 'vit_tiny_patch16_224'
    
    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    PRETRAIN_EPOCHS = 10 # Keep low for a quick demo
    FINETUNE_EPOCHS = 10 # Keep low for a quick demo
    LR = 1e-4
    
    # Fine-tuning Strategies
    L2_SP_LAMBDA = 0.0001
    DISTILL_LAMBDA = 1.0
    DISTILL_TEMP = 1.0
    EMA_ALPHA = 0.999

    # Initialization Options
    # Set to True to randomly initialize encoders instead of using pretrained weights
    RANDOM_INIT_IMAGE = False
    RANDOM_INIT_TEXT = False

# 2. Data Handling

# 2a. Colored MNIST Dataset with Spurious Correlation
class ColoredMNIST(Dataset):
    def __init__(self, mnist_dataset, correlation=0.99, background_alpha=0.7):
        self.mnist_dataset = mnist_dataset
        self.correlation = correlation
        self.num_classes = 10
        self.background_alpha = background_alpha
        # Define a color for each digit
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 128)
        ]

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx] # image is single channel
        
        # Apply spurious correlation
        if random.random() < self.correlation:
            color = self.colors[label]
        else:
            color = random.choice(self.colors)
        
        # Build background as a light (pastel) version of the digit color
        # Blend with white using background_alpha
        bg_color = tuple(int(color[i] * (1 - self.background_alpha) + 255 * self.background_alpha) for i in range(3))
        
        # Create mask from the grayscale image
        gray_np = np.array(image)  # (H, W)
        mask = gray_np > 50  # True for digit strokes
        
        # Create RGB image initialized to background color, then paint digit with full color
        h, w = gray_np.shape
        colored_image_np = np.zeros((h, w, 3), dtype=np.uint8)
        colored_image_np[:] = bg_color
        for i in range(3):
            channel = colored_image_np[..., i]
            channel[mask] = color[i]
            colored_image_np[..., i] = channel
            
        return transforms.ToPILImage()(colored_image_np), label

# 2b. Captioning Wrapper Dataset
class CaptionedDataset(Dataset):
    def __init__(self, base_dataset, captions_templates, tokenizer, max_len=32):
        self.base_dataset = base_dataset
        self.captions = captions_templates
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Original MNIST is single channel, convert to 3-channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        
        caption = random.choice(self.captions).format(label)
        
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        return image, tokens, label

# 3. Model Architecture
class ImageEncoder(nn.Module):
    def __init__(self, model_name=Config.VISION_MODEL_NAME, pretrained=None):
        super().__init__()
        if pretrained is None:
            pretrained = not Config.RANDOM_INIT_IMAGE
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # num_classes=0 to get feature vector
        self.projection = nn.Linear(self.model.num_features, Config.EMBED_DIM)

    def forward(self, x):
        features = self.model(x)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, model_name=Config.TEXT_MODEL_NAME, pretrained=None):
        super().__init__()
        if pretrained is None:
            pretrained = not Config.RANDOM_INIT_TEXT
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
        self.projection = nn.Linear(self.model.config.hidden_size, Config.EMBED_DIM)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding
        last_hidden_state = output.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return F.normalize(projected, p=2, dim=-1)

class CLIPModel(nn.Module):
    def __init__(self, pretrained_image=None, pretrained_text=None):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image)
        self.text_encoder = TextEncoder(pretrained=pretrained_text)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, text_tokens):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(text_tokens['input_ids'].squeeze(1), text_tokens['attention_mask'].squeeze(1))
        return image_embeds, text_embeds, self.logit_scale.exp()

# 4. Contrastive Loss and Evaluation
def contrastive_loss(image_embeds, text_embeds, logit_scale):
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

@torch.no_grad()
def get_text_prototypes(text_encoder, tokenizer):
    """
    Generates average text embeddings for each digit (0-9) to use as classifiers.
    """
    prototypes = {}
    text_encoder.eval()
    for i in range(10):
        captions = [template.format(i) for template in Config.CAPTIONS]
        tokens = tokenizer(
            captions, padding=True, return_tensors="pt"
        ).to(Config.DEVICE)
        
        embeds = text_encoder(tokens['input_ids'], tokens['attention_mask'])
        prototypes[i] = embeds.mean(dim=0)
    
    # Stack prototypes into a single tensor
    prototype_tensor = torch.stack([prototypes[i] for i in range(10)])
    return F.normalize(prototype_tensor, p=2, dim=-1)

@torch.no_grad()
def evaluate(model, dataloader, text_prototypes):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    for batch in dataloader:
        images, texts, labels = batch
        images = images.to(Config.DEVICE)
        for k, v in texts.items():
            texts[k] = v.to(Config.DEVICE)
        
        image_embeds, text_embeds, logit_scale = model(images, texts)
        loss = contrastive_loss(image_embeds, text_embeds, logit_scale)
        total_loss += loss.item()
        
        # Retrieval-based classification
        similarities = image_embeds @ text_prototypes.t() # (B, 10)
        
        # Get top-k predictions
        _, topk_preds = torch.topk(similarities, k=3, dim=-1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(topk_preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    top1_acc = np.mean(all_labels == all_preds[:, 0])
    top2_acc = np.mean([l in p for l, p in zip(all_labels, all_preds[:, :2])])
    top3_acc = np.mean([l in p for l, p in zip(all_labels, all_preds[:, :3])])
    
    return {
        "loss": avg_loss,
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "top3_acc": top3_acc,
    }

# 5. Training Loop
def train(model, train_loader, val_loader, optimizer, epochs, text_prototypes, strategy="direct", pretrained_model=None):
    if strategy == "dynamic_distill":
        # Create an EMA teacher model
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
    elif strategy in ["static_distill", "l2_sp"]:
        assert pretrained_model is not None
        pretrained_model.eval()

    for epoch in range(epochs):
        model.train()
        if strategy != "direct":
            print(f"Fine-tuning with strategy: {strategy}")
        
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            images, texts, _ = batch
            images = images.to(Config.DEVICE)
            for k, v in texts.items():
                texts[k] = v.to(Config.DEVICE)
            
            image_embeds, text_embeds, logit_scale = model(images, texts)
            
            # Main contrastive loss
            loss = contrastive_loss(image_embeds, text_embeds, logit_scale)
            pbar.set_postfix(loss=loss.item())
            
            # --- Strategy-specific regularization ---
            if strategy == "l2_sp":
                l2_reg = 0.0
                for p_finetuned, p_pretrained in zip(model.image_encoder.parameters(), pretrained_model.image_encoder.parameters()):
                    l2_reg += (p_finetuned - p_pretrained).pow(2).sum()
                loss += Config.L2_SP_LAMBDA * l2_reg
                
            elif strategy in ["static_distill", "dynamic_distill"]:
                teacher = teacher_model if strategy == "dynamic_distill" else pretrained_model
                with torch.no_grad():
                    teacher_image_embeds, _, _ = teacher(images, texts)
                
                # Distillation loss (KL divergence on similarity scores)
                student_sim = image_embeds @ text_embeds.t()
                teacher_sim = teacher_image_embeds @ text_embeds.t()

                distill_loss = F.kl_div(
                    F.log_softmax(student_sim / Config.DISTILL_TEMP, dim=1),
                    F.softmax(teacher_sim / Config.DISTILL_TEMP, dim=1),
                    reduction='batchmean'
                ) * (Config.DISTILL_TEMP ** 2)
                loss += distill_loss * Config.DISTILL_LAMBDA
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Update EMA teacher model if using dynamic distillation
            if strategy == "dynamic_distill":
                with torch.no_grad():
                    for student_p, teacher_p in zip(model.parameters(), teacher_model.parameters()):
                        teacher_p.data.mul_(Config.EMA_ALPHA).add_(student_p.data, alpha=1 - Config.EMA_ALPHA)
                        
        avg_train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, text_prototypes)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Top-1 Acc: {val_metrics['top1_acc']:.4f}")

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-init-image', action='store_true', help='Randomly initialize image encoder')
    parser.add_argument('--random-init-text', action='store_true', help='Randomly initialize text encoder')
    parser.add_argument('--random-init-both', action='store_true', help='Randomly initialize both encoders')
    args = parser.parse_args()

    if args.random_init_both:
        Config.RANDOM_INIT_IMAGE = True
        Config.RANDOM_INIT_TEXT = True
    else:
        if args.random_init_image:
            Config.RANDOM_INIT_IMAGE = True
        if args.random_init_text:
            Config.RANDOM_INIT_TEXT = True

    print(f"Using device: {Config.DEVICE}")
    print(f"Initialization - image: {'random' if Config.RANDOM_INIT_IMAGE else 'pretrained'}, text: {'random' if Config.RANDOM_INIT_TEXT else 'pretrained'}")
    
    # --- Prepare Data ---
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    
    # Pre-training data (MNIST)
    mnist_train_full = MNIST(root='./data', train=True, download=True, transform=None)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=None)
    
    # Create a small validation set from the training data
    train_size = int(0.9 * len(mnist_train_full))
    val_size = len(mnist_train_full) - train_size
    mnist_train, mnist_val = random_split(mnist_train_full, [train_size, val_size])

    pretrain_train_dataset = CaptionedDataset(mnist_train, Config.CAPTIONS, tokenizer)
    pretrain_val_dataset = CaptionedDataset(mnist_val, Config.CAPTIONS, tokenizer)
    pretrain_test_dataset = CaptionedDataset(mnist_test, Config.CAPTIONS, tokenizer)

    pretrain_train_loader = DataLoader(pretrain_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=Config.BATCH_SIZE)
    pretrain_test_loader = DataLoader(pretrain_test_dataset, batch_size=Config.BATCH_SIZE)

    # Fine-tuning data (Colored MNIST)
    colored_mnist_train = ColoredMNIST(mnist_train)
    colored_mnist_val = ColoredMNIST(mnist_val)
    colored_mnist_test = ColoredMNIST(mnist_test)
    
    finetune_train_dataset = CaptionedDataset(colored_mnist_train, Config.CAPTIONS, tokenizer)
    finetune_val_dataset = CaptionedDataset(colored_mnist_val, Config.CAPTIONS, tokenizer)
    finetune_test_dataset = CaptionedDataset(colored_mnist_test, Config.CAPTIONS, tokenizer)

    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=Config.BATCH_SIZE)
    finetune_test_loader = DataLoader(finetune_test_dataset, batch_size=Config.BATCH_SIZE)
    
    results = {}

    # --- 6. Pre-training ---
    print("\n--- Starting Pre-training ---")
    pretrained_model = CLIPModel().to(Config.DEVICE)
    optimizer = optim.Adam(pretrained_model.parameters(), lr=Config.LR)
    
    # We need prototypes for validation during training
    text_prototypes_pre = get_text_prototypes(pretrained_model.text_encoder, tokenizer)
    
    train(pretrained_model, pretrain_train_loader, pretrain_val_loader, optimizer, Config.PRETRAIN_EPOCHS, text_prototypes_pre)
    
    print("\n--- Evaluating Pre-trained Model ---")
    results['pretrained'] = {}
    # Evaluate on pre-training data splits
    results['pretrained']['pretrain_train'] = evaluate(pretrained_model, pretrain_train_loader, text_prototypes_pre)
    results['pretrained']['pretrain_val'] = evaluate(pretrained_model, pretrain_val_loader, text_prototypes_pre)
    results['pretrained']['pretrain_test'] = evaluate(pretrained_model, pretrain_test_loader, text_prototypes_pre)
    # Evaluate on fine-tuning data splits (zero-shot)
    results['pretrained']['finetune_train'] = evaluate(pretrained_model, finetune_train_loader, text_prototypes_pre)
    results['pretrained']['finetune_val'] = evaluate(pretrained_model, finetune_val_loader, text_prototypes_pre)
    results['pretrained']['finetune_test'] = evaluate(pretrained_model, finetune_test_loader, text_prototypes_pre)

    # Save pre-trained state for re-use
    pretrained_state_dict = copy.deepcopy(pretrained_model.state_dict())
    
    # --- 8. Fine-tuning with Four Strategies ---
    finetune_strategies = ["direct", "l2_sp", "static_distill", "dynamic_distill"]
    
    for strategy in finetune_strategies:
        print(f"\n--- Starting Fine-tuning with strategy: {strategy} ---")
        
        # Load a fresh copy of the pre-trained model for each strategy
        model_to_finetune = CLIPModel().to(Config.DEVICE)
        model_to_finetune.load_state_dict(pretrained_state_dict)
        
        # Freeze the text encoder for ALL fine-tuning strategies
        for param in model_to_finetune.text_encoder.parameters():
            param.requires_grad = False
            
        # Only image encoder parameters will be optimized
        optimizer = optim.Adam(model_to_finetune.image_encoder.parameters(), lr=Config.LR / 10) # Lower LR for fine-tuning
        
        # The text prototypes are fixed since the text encoder is frozen
        text_prototypes_ft = get_text_prototypes(model_to_finetune.text_encoder, tokenizer)
        
        train(model_to_finetune, finetune_train_loader, finetune_val_loader, optimizer, 
              Config.FINETUNE_EPOCHS, text_prototypes_ft, strategy=strategy, pretrained_model=pretrained_model)
              
        print(f"\n--- Evaluating Fine-tuned Model ({strategy}) ---")
        results[strategy] = {}
        # Evaluate on pre-training data splits (to check for catastrophic forgetting)
        results[strategy]['pretrain_train'] = evaluate(model_to_finetune, pretrain_train_loader, text_prototypes_ft)
        results[strategy]['pretrain_val'] = evaluate(model_to_finetune, pretrain_val_loader, text_prototypes_ft)
        results[strategy]['pretrain_test'] = evaluate(model_to_finetune, pretrain_test_loader, text_prototypes_ft)
        # Evaluate on fine-tuning data splits
        results[strategy]['finetune_train'] = evaluate(model_to_finetune, finetune_train_loader, text_prototypes_ft)
        results[strategy]['finetune_val'] = evaluate(model_to_finetune, finetune_val_loader, text_prototypes_ft)
        results[strategy]['finetune_test'] = evaluate(model_to_finetune, finetune_test_loader, text_prototypes_ft)

    # --- 9. Report All Metrics ---
    print("\n\n--- FINAL RESULTS SUMMARY ---")
    
    report_data = []
    for model_name, model_results in results.items():
        for split_name, metrics in model_results.items():
            row = {
                'Model': model_name,
                'Split': split_name,
                'Loss': f"{metrics['loss']:.4f}",
                'Top-1 Acc': f"{metrics['top1_acc']:.3f}",
                'Top-2 Acc': f"{metrics['top2_acc']:.3f}",
                'Top-3 Acc': f"{metrics['top3_acc']:.3f}",
            }
            report_data.append(row)
            
    df = pd.DataFrame(report_data)
    print(df.to_string())