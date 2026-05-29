#!/usr/bin/env python3
"""
CLIP Fine-tuning Pipeline for Cassava Disease Classification
Trains 3 scenarios: 10 epochs, 50 epochs, and 50 epochs low LR
Uses 43 images per class with seed 42 for reproducibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
import pandas as pd
import os
import random
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse
from datetime import datetime

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
DATASET_PATH = 'data/CD1/Images'
OUTPUT_DIR = 'results/clip_finetuning'
MODEL_NAME = 'openai/clip-vit-base-patch16'
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGES_PER_CLASS = 43

# Class configuration
CLASS_NAMES = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
CLASS_PROMPTS = {
    'CBB': 'a photo of a cassava leaf with bacterial blight',
    'CBSD': 'a photo of a cassava leaf with brown streak disease',
    'CGM': 'a photo of a cassava leaf with green mottle', 
    'CMD': 'a photo of a cassava leaf with mosaic disease',
    'Healthy': 'A photo of a healthy cassava leaf'
}

# Training scenarios
TRAINING_SCENARIOS = {
    'scenario_1': {'epochs': 10, 'lr': 1e-5, 'name': '10_epochs'},
    'scenario_2': {'epochs': 50, 'lr': 1e-5, 'name': '50_epochs'},
    'scenario_3': {'epochs': 50, 'lr': 1e-6, 'name': '50_epochs_low_lr'}
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CLIP fine-tuning experiments on CD1.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to CD1 Images directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for trained checkpoints.")
    parser.add_argument("--images-per-class", type=int, default=IMAGES_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_train_test_split(dataset_path, images_per_class=43, seed=42):
    """Create train/test split with specified images per class for training"""
    print(f"Creating train/test split with {images_per_class} images per class for training (seed={seed})...")
    
    # Set local random state for consistent sampling
    local_random = random.Random(seed)
    
    train_files = {}
    test_files = {}
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            # Get all image files
            all_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle files with seed
            local_random.shuffle(all_files)
            
            # Split into train and test
            if len(all_files) >= images_per_class:
                train_files[class_name] = all_files[:images_per_class]
                test_files[class_name] = all_files[images_per_class:]
            else:
                print(f"Warning: {class_name} has only {len(all_files)} images, using all for training")
                train_files[class_name] = all_files
                test_files[class_name] = []
            
            print(f"  {class_name}: {len(train_files[class_name])} train, {len(test_files[class_name])} test")
    
    return train_files, test_files

class CassavaDataset(Dataset):
    """General dataset class for both train and test"""
    
    def __init__(self, dataset_path, file_dict, transform, split_name=""):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
        
        total_images = 0
        for class_name in CLASS_NAMES:
            if class_name in file_dict:
                class_dir = os.path.join(dataset_path, class_name)
                for filename in file_dict[class_name]:
                    self.samples.append({
                        'path': os.path.join(class_dir, filename),
                        'class_idx': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
                total_images += len(file_dict[class_name])
        
        print(f"{split_name} dataset created: {total_images} total images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        return {
            'image': self.transform(image),
            'class_idx': sample['class_idx'],
            'class_name': sample['class_name']
        }

class FineTunedCLIP(nn.Module):
    """CLIP model with classification head for fine-tuning"""
    
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Fully freeze text encoder (text model + text projection)
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.text_projection.parameters():
            param.requires_grad = False
        
        print("✅ Text encoder fully frozen (text_model + text_projection)")
        
        # Classification head
        self.classifier = nn.Linear(self.clip_model.config.projection_dim, num_classes)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, images, return_features=False):
        # Get image features from CLIP
        image_features = self.clip_model.get_image_features(images)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Classification
        logits = self.classifier(image_features)
        
        if return_features:
            return logits, image_features
        return logits

def create_transforms():
    """Create training and evaluation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    return train_transform, eval_transform

def train_model(model, dataloader, epochs, learning_rate, scenario_name):
    """Train the model for specified epochs and learning rate"""
    print(f"\nTraining {scenario_name}: {epochs} epochs, LR={learning_rate}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_steps = len(dataloader) * epochs
    step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            labels = batch['class_idx'].to(DEVICE)
            
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            step += 1
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    print(f"Training {scenario_name} completed!")
    return model

def evaluate_model(model, dataloader):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(DEVICE)
            labels = batch['class_idx']
            
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_true_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_true_labels)

def calculate_metrics(predictions, true_labels):
    """Calculate the 5 required metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    return {
        'Accuracy': accuracy,
        'F1-Score (Macro)': f1_macro,
        'F1-Score (Weighted)': f1_weighted,
        'Precision (Weighted)': precision_weighted,
        'Recall (Weighted)': recall_weighted
    }

def run_zero_shot_baseline(test_dataloader):
    """Run zero-shot evaluation on test set as baseline"""
    print("Running Zero-Shot CLIP baseline on test set...")
    
    # Load original CLIP model
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Precompute text features
    prompts = [CLASS_PROMPTS[cls] for cls in CLASS_NAMES]
    with torch.no_grad():
        text_inputs = tokenizer(prompts, padding=True, truncation=True,
                               max_length=77, return_tensors="pt").to(DEVICE)
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
    
    # Evaluate on test set
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Zero-shot evaluation"):
            images = batch['image'].to(DEVICE)
            labels = batch['class_idx']
            
            # Get image features
            image_features = model.get_image_features(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Calculate similarities and predictions
            similarities = torch.matmul(image_features, text_features.T)
            predictions = similarities.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_true_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_true_labels)

def main():
    args = parse_args()
    global DATASET_PATH, OUTPUT_DIR, BATCH_SIZE, IMAGES_PER_CLASS, SEED
    DATASET_PATH = args.dataset_path
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    IMAGES_PER_CLASS = args.images_per_class
    SEED = args.seed
    set_seed(SEED)

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_PATH}. "
            "Place CD1 images under data/CD1/Images or pass --dataset-path."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    """Main training and evaluation pipeline"""
    print("CLIP Fine-tuning Pipeline")
    print(f"Seed: {SEED}")
    print(f"Images per class for training: {IMAGES_PER_CLASS}")
    print(f"Device: {DEVICE}")
    
    # Create train/test split
    train_files, test_files = create_train_test_split(DATASET_PATH, IMAGES_PER_CLASS, SEED)
    
    # Create transforms
    train_transform, eval_transform = create_transforms()
    
    # Create datasets
    train_dataset = CassavaDataset(DATASET_PATH, train_files, train_transform, "Training")
    test_dataset = CassavaDataset(DATASET_PATH, test_files, eval_transform, "Test")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    
    print(f"\nDataset Summary:")
    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    
    # Store all results
    all_results = {}
    
    # 1. Zero-shot baseline on test set
    print("\n" + "="*60)
    print("1. ZERO-SHOT BASELINE (Test Set)")
    print("="*60)
    zero_predictions, zero_labels = run_zero_shot_baseline(test_dataloader)
    zero_metrics = calculate_metrics(zero_predictions, zero_labels)
    all_results['zero_shot'] = zero_metrics
    
    print("Zero-shot Results:")
    for metric, value in zero_metrics.items():
        print(f"  {metric:<25}: {value:.4f}")
    
    # 2. Fine-tuned scenarios
    for scenario_key, config in TRAINING_SCENARIOS.items():
        print("\n" + "="*60)
        print(f"2. FINE-TUNING: {config['name'].upper()}")
        print("="*60)
        
        # Create fresh model for each scenario
        model = FineTunedCLIP(MODEL_NAME, len(CLASS_NAMES)).to(DEVICE)
        
        # Train model on training set
        trained_model = train_model(
            model, train_dataloader, 
            config['epochs'], config['lr'], 
            config['name']
        )
        
        # Evaluate on test set
        predictions, true_labels = evaluate_model(trained_model, test_dataloader)
        metrics = calculate_metrics(predictions, true_labels)
        all_results[scenario_key] = metrics
        
        print(f"\n{config['name']} Results (Test Set):")
        for metric, value in metrics.items():
            print(f"  {metric:<25}: {value:.4f}")
        
        # Save model
        model_path = os.path.join(args.checkpoint_dir, f"clip_finetuned_{config['name']}.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': config,
            'metrics': metrics,
            'class_names': CLASS_NAMES
        }, model_path)
        print(f"Model saved: {model_path}")
    
    # 3. Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL SCENARIOS (Test Set Evaluation)")
    print("="*80)
    
    # Create comparison table
    metrics_names = ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)']
    scenario_names = ['Zero-Shot', '10 Epochs', '50 Epochs', '50 Epochs Low LR']
    scenario_keys = ['zero_shot', 'scenario_1', 'scenario_2', 'scenario_3']
    
    print(f"{'Scenario':<20}", end="")
    for metric in metrics_names:
        print(f"{metric:<20}", end="")
    print()
    print("-" * (20 + 20 * len(metrics_names)))
    
    for i, (scenario_key, scenario_name) in enumerate(zip(scenario_keys, scenario_names)):
        print(f"{scenario_name:<20}", end="")
        for metric in metrics_names:
            value = all_results[scenario_key][metric]
            print(f"{value:<20.4f}", end="")
        print()
    
    # Save results to JSON
    results_summary = {
        'experiment_info': {
            'seed': SEED,
            'images_per_class_training': IMAGES_PER_CLASS,
            'total_training_images': len(train_dataset),
            'total_test_images': len(test_dataset),
            'model_name': MODEL_NAME,
            'training_scenarios': TRAINING_SCENARIOS,
            'evaluation_note': 'All evaluations performed on test set (unseen data)',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'dataset_split': {
            'train_files_per_class': {cls: len(train_files[cls]) for cls in CLASS_NAMES},
            'test_files_per_class': {cls: len(test_files[cls]) for cls in CLASS_NAMES}
        },
        'results': all_results
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'clip_finetuning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("All models saved as .pth files")
    print(f"\nNote: All evaluations performed on {len(test_dataset)} test images (unseen during training)")
    
    return all_results

if __name__ == "__main__":
    results = main()
