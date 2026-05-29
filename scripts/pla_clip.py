#!/usr/bin/env python3
"""
PLA-CLIP training for few-shot cassava disease recognition.

This script implements the paper-facing Progressive Layer Activation CLIP setup:
- CLIP ViT-B/16 backbone
- text encoder frozen
- vision transformer layers progressively activated at epochs 0, 10, 20, and 30
- CLIP-style symmetric contrastive loss
- default few-shot split of 43 images per class with a fixed seed
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


SEED = 43
MODEL_NAME = "openai/clip-vit-base-patch16"
DATASET_PATH = "data/CD1/Images"
OUTPUT_DIR = "results/pla_clip"
CHECKPOINT_DIR = "checkpoints"
IMAGES_PER_CLASS = 43
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
CLASS_PROMPTS = {
    "CBB": "a photo of a cassava leaf with bacterial blight",
    "CBSD": "a photo of a cassava leaf with brown streak disease",
    "CGM": "a photo of a cassava leaf with green mottle",
    "CMD": "a photo of a cassava leaf with mosaic disease",
    "Healthy": "a photo of a healthy cassava leaf",
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train PLA-CLIP on CD1.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to CD1 Images directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for JSON reports.")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="Directory for trained checkpoints.")
    parser.add_argument("--images-per-class", type=int, default=IMAGES_PER_CLASS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def create_train_test_split(dataset_path, images_per_class, seed):
    local_random = random.Random(seed)
    train_files = {}
    test_files = {}

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        all_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        local_random.shuffle(all_files)

        train_files[class_name] = all_files[:images_per_class]
        test_files[class_name] = all_files[images_per_class:]
        print(f"{class_name}: {len(train_files[class_name])} train, {len(test_files[class_name])} test")

    return train_files, test_files


class CassavaDataset(Dataset):
    def __init__(self, dataset_path, files_by_class, transform):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(dataset_path, class_name)
            for filename in files_by_class[class_name]:
                self.samples.append({
                    "path": os.path.join(class_dir, filename),
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "prompt": CLASS_PROMPTS[class_name],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "class_idx": sample["class_idx"],
            "prompt": sample["prompt"],
        }


def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    return train_transform, eval_transform


def active_layers_for_epoch(epoch):
    if epoch < 10:
        return list(range(9, 12))
    if epoch < 20:
        return list(range(6, 12))
    if epoch < 30:
        return list(range(3, 12))
    return list(range(0, 12))


def apply_progressive_activation(model, epoch):
    active_layers = active_layers_for_epoch(epoch)

    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in active_layers:
        for param in model.vision_model.encoder.layers[layer_idx].parameters():
            param.requires_grad = True

    # Keep projection/normalization trainable while the text encoder remains frozen.
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True
    model.logit_scale.requires_grad = True

    return active_layers


def contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * image_features @ text_features.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_loss = F.cross_entropy(logits, labels)
    text_loss = F.cross_entropy(logits.t(), labels)
    return (image_loss + text_loss) / 2


def train_one_epoch(model, tokenizer, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(DEVICE)
        prompts = batch["prompt"]
        text_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(DEVICE)

        optimizer.zero_grad()
        image_features = model.get_image_features(pixel_values=images)
        text_features = model.get_text_features(**text_inputs)
        loss = contrastive_loss(image_features, text_features, model.logit_scale)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, tokenizer, dataloader):
    model.eval()
    prompts = [CLASS_PROMPTS[class_name] for class_name in CLASS_NAMES]
    text_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(DEVICE)
    text_features = F.normalize(model.get_text_features(**text_inputs), dim=-1)

    predictions = []
    labels = []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(DEVICE)
        image_features = F.normalize(model.get_image_features(pixel_values=images), dim=-1)
        logits = model.logit_scale.exp() * image_features @ text_features.t()
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(batch["class_idx"].numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "F1-Score (Macro)": f1_score(labels, predictions, average="macro", zero_division=0),
        "F1-Score (Weighted)": f1_score(labels, predictions, average="weighted", zero_division=0),
        "Precision (Weighted)": precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {args.dataset_path}. "
            "Place CD1 images under data/CD1/Images or pass --dataset-path."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("PLA-CLIP training")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Seed: {args.seed}")
    print(f"Device: {DEVICE}")

    train_files, test_files = create_train_test_split(
        args.dataset_path,
        args.images_per_class,
        args.seed,
    )
    train_transform, eval_transform = create_transforms()
    train_dataset = CassavaDataset(args.dataset_path, train_files, train_transform)
    test_dataset = CassavaDataset(args.dataset_path, test_files, eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    history = []
    for epoch in range(args.epochs):
        active_layers = apply_progressive_activation(model, epoch)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        loss = train_one_epoch(model, tokenizer, train_loader, optimizer)
        metrics = evaluate(model, tokenizer, test_loader)
        history.append({
            "epoch": epoch + 1,
            "active_layers": active_layers,
            "loss": loss,
            "metrics": metrics,
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"layers={active_layers} | loss={loss:.4f} | "
            f"acc={metrics['Accuracy']:.4f} | f1w={metrics['F1-Score (Weighted)']:.4f}"
        )

    final_metrics = history[-1]["metrics"] if history else {}
    checkpoint_path = os.path.join(args.checkpoint_dir, "pla_clip.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "class_names": CLASS_NAMES,
        "class_prompts": CLASS_PROMPTS,
        "seed": args.seed,
        "images_per_class": args.images_per_class,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "final_metrics": final_metrics,
    }, checkpoint_path)

    report = {
        "experiment": "PLA-CLIP",
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "dataset_path": args.dataset_path,
        "images_per_class": args.images_per_class,
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "activation_schedule": {
            "epochs_1_10": "T9-T11",
            "epochs_11_20": "T6-T11",
            "epochs_21_30": "T3-T11",
            "epochs_31_plus": "T0-T11",
        },
        "final_metrics": final_metrics,
        "history": history,
        "checkpoint": checkpoint_path,
    }
    report_path = os.path.join(args.output_dir, "pla_clip_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
