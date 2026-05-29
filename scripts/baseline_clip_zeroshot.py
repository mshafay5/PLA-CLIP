#!/usr/bin/env python3
"""
Zero-Shot CLIP Evaluation for Multiple Cassava Disease Classification Datasets
Evaluates CD1, CD2, and CD3 datasets
Calculates: Accuracy, F1-Score (Macro), F1-Score (Weighted), Precision (Weighted), Recall (Weighted)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
import pandas as pd
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from datetime import datetime

# Configuration
MODEL_NAME = 'openai/clip-vit-base-patch16'
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset configurations
DATASET_CONFIGS = {
    'CD1': {
        'path': 'data/CD1/Images',
        'class_names': ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy'],
        'class_prompts': {
            'CBB': 'a photo of a cassava leaf with bacterial blight',
            'CBSD': 'a photo of a cassava leaf with brown streak disease',
            'CGM': 'a photo of a cassava leaf with green mottle', 
            'CMD': 'a photo of a cassava leaf with mosaic disease',
            'Healthy': 'A photo of a healthy cassava leaf'
        }
    },
    'CD2': {
        'path': 'data/CD2/Images',
        'class_names': ['CBSD','CMD', 'Healthy'],
        'class_prompts': {
            'CBSD': 'a photo of a cassava leaf with brown streak disease',
            'CMD': 'a photo of a cassava leaf with mosaic disease',
            'Healthy': 'A photo of a healthy cassava leaf'
        }
    },
    'CD3': {
        'path': 'data/CD3/Images',
        'class_names': ['CRRD','CMD', 'Healthy', 'CBB', 'CBLS'],
        'class_prompts': {
            'CBB': 'a photo of a cassava leaf with bacterial blight',
            'CRRD': 'a photo of a cassava leaf with root rot disease',
            'CBLS': 'a photo of a cassava leaf with brown leaf spot', 
            'CMD': 'a photo of a cassava leaf with mosaic disease',
            'Healthy': 'A photo of a healthy cassava leaf'
        }
    }
}

class CassavaDataset(Dataset):
    """Dataset for cassava images with configurable classes"""
    
    def __init__(self, dataset_path, class_names, transform):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.class_names = class_names
        
        total_images = 0
        for class_name in class_names:
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for filename in image_files:
                    self.samples.append({
                        'path': os.path.join(class_dir, filename),
                        'class_idx': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
                
                total_images += len(image_files)
                print(f"  {class_name}: {len(image_files)} images")
            else:
                print(f"  Warning: {class_dir} does not exist!")
        
        print(f"  Total: {total_images} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
            return {
                'image': self.transform(image),
                'class_idx': sample['class_idx'],
                'class_name': sample['class_name']
            }
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a random other sample
            return self.__getitem__((idx + 1) % len(self.samples))

def create_transform():
    """Create image transformation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])

def load_clip_model():
    """Load CLIP model"""
    print(f"Loading CLIP model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer

def get_text_features(model, tokenizer, class_prompts):
    """Get text features for given prompts"""
    prompts = list(class_prompts.values())
    with torch.no_grad():
        text_inputs = tokenizer(prompts, padding=True, truncation=True,
                               max_length=77, return_tensors="pt").to(DEVICE)
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features

def evaluate_dataset(model, tokenizer, dataset_config, dataset_name):
    """Evaluate model on a single dataset"""
    print(f"\nEvaluating {dataset_name}:")
    print("-" * 40)
    
    # Check if dataset path exists
    if not os.path.exists(dataset_config['path']):
        print(f"❌ Dataset path does not exist: {dataset_config['path']}")
        return None
    
    # Create dataset
    transform = create_transform()
    dataset = CassavaDataset(
        dataset_config['path'], 
        dataset_config['class_names'], 
        transform
    )
    
    if len(dataset) == 0:
        print("❌ No images found in dataset")
        return None
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Get text features for this dataset's classes
    text_features = get_text_features(model, tokenizer, dataset_config['class_prompts'])
    
    # Run evaluation
    all_predictions = []
    all_true_labels = []
    
    print("Running zero-shot evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(DEVICE)
            true_labels = batch['class_idx'].cpu().numpy()
            
            # Get image features
            image_features = model.get_image_features(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Calculate similarities and get predictions
            similarities = torch.matmul(image_features, text_features.T)
            predictions = similarities.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_true_labels)
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<25}: {value:.4f} ({100*value:.2f}%)")
    
    # Calculate confusion matrix and per-class metrics
    cm = confusion_matrix(all_true_labels, all_predictions)
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=dataset_config['class_names'],
        zero_division=0,
        output_dict=True
    )
    
    return {
        'dataset_name': dataset_name,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'num_samples': len(dataset),
        'class_names': dataset_config['class_names'],
        'num_classes': len(dataset_config['class_names'])
    }

def calculate_metrics(predictions, true_labels):
    """Calculate the 5 required metrics"""
    if len(predictions) == 0 or len(true_labels) == 0:
        return {
            'Accuracy': 0.0,
            'F1-Score (Macro)': 0.0,
            'F1-Score (Weighted)': 0.0,
            'Precision (Weighted)': 0.0,
            'Recall (Weighted)': 0.0
        }
    
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

def create_comparison_table(all_results):
    """Create comparison table of results"""
    if not all_results:
        return None
    
    print("\n" + "="*80)
    print("ZERO-SHOT CLIP RESULTS COMPARISON")
    print("="*80)
    
    # Create comparison data
    comparison_data = []
    for result in all_results:
        if result:  # Skip None results
            row = {
                'Dataset': result['dataset_name'],
                'Classes': result['num_classes'],
                'Samples': result['num_samples'],
                'Accuracy': f"{result['metrics']['Accuracy']:.4f}",
                'F1-Macro': f"{result['metrics']['F1-Score (Macro)']:.4f}",
                'F1-Weighted': f"{result['metrics']['F1-Score (Weighted)']:.4f}",
                'Precision (W)': f"{result['metrics']['Precision (Weighted)']:.4f}",
                'Recall (W)': f"{result['metrics']['Recall (Weighted)']:.4f}"
            }
            comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Find best performing dataset
        best_accuracy = max([float(row['Accuracy']) for row in comparison_data])
        best_dataset = next(row['Dataset'] for row in comparison_data if float(row['Accuracy']) == best_accuracy)
        
        print(f"\n🏆 Best Performance: {best_dataset} with {best_accuracy:.4f} accuracy")
        
        return df
    
    return None

def save_results(all_results, comparison_df, output_dir='results/zero_shot_clip'):
    """Save results to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f'zero_shot_clip_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'method': 'Zero-Shot CLIP',
            'datasets_evaluated': [r['dataset_name'] for r in all_results if r],
            'results': [r for r in all_results if r]
        }, f, indent=2)
    
    # Save comparison table
    if comparison_df is not None:
        csv_file = os.path.join(output_dir, f'zero_shot_clip_comparison_{timestamp}.csv')
        comparison_df.to_csv(csv_file, index=False)
        print(f"\n✅ Results saved:")
        print(f"  - Detailed: {results_file}")
        print(f"  - Summary: {csv_file}")
        return results_file, csv_file
    
    return results_file, None

def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot CLIP evaluation on CD1/CD2/CD3.")
    parser.add_argument("--data-root", default="data", help="Root directory containing CD1, CD2, and CD3 folders.")
    parser.add_argument("--datasets", nargs="+", default=["CD1", "CD2", "CD3"], choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", default="results/zero_shot_clip", help="Directory for evaluation reports.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size.")
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    for dataset_name, config in DATASET_CONFIGS.items():
        config["path"] = os.path.join(args.data_root, dataset_name, "Images")

    print("Zero-Shot CLIP Multi-Dataset Evaluation")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
    print("="*60)
    
    # Load model once
    model, tokenizer = load_clip_model()
    
    # Evaluate all datasets
    all_results = []
    datasets_to_evaluate = args.datasets
    
    for dataset_name in datasets_to_evaluate:
        if dataset_name in DATASET_CONFIGS:
            result = evaluate_dataset(model, tokenizer, DATASET_CONFIGS[dataset_name], dataset_name)
            all_results.append(result)
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            all_results.append(None)
    
    # Create comparison table
    comparison_df = create_comparison_table(all_results)
    
    # Save results
    results_file, csv_file = save_results(all_results, comparison_df, args.output_dir)
    
    # Performance analysis
    valid_results = [r for r in all_results if r]
    if valid_results:
        print(f"\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Sort by accuracy
        sorted_results = sorted(valid_results, key=lambda x: x['metrics']['Accuracy'], reverse=True)
        
        print("Ranking by Accuracy:")
        for i, result in enumerate(sorted_results, 1):
            accuracy = result['metrics']['Accuracy']
            dataset = result['dataset_name']
            classes = result['num_classes']
            samples = result['num_samples']
            print(f"  {i}. {dataset}: {accuracy:.4f} ({classes} classes, {samples:,} samples)")
        
        # Dataset complexity analysis
        print(f"\nDataset Complexity Analysis:")
        for result in sorted_results:
            dataset = result['dataset_name']
            classes = result['num_classes']
            samples = result['num_samples']
            accuracy = result['metrics']['Accuracy']
            samples_per_class = samples // classes
            print(f"  {dataset}: {accuracy:.4f} accuracy | {classes} classes | {samples_per_class:,} samples/class")
    
    print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_results

if __name__ == "__main__":
    results = main()
