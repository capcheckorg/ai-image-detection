#!/usr/bin/env python3
"""
CapCheck AI Detection - CLI Training Script

Fine-tune the ViT model for AI image detection from the command line.
This is an alternative to the Jupyter notebook for automated/scripted training.

Usage:
    python train_vit.py --data-dir ./data --output-dir ./checkpoints

    # With custom hyperparameters
    python train_vit.py --data-dir ./data --epochs 5 --batch-size 8 --lr 1e-5

    # Push to HuggingFace after training
    python train_vit.py --data-dir ./data --push-to-hub --hf-repo capcheck/ai-image-detection
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS")
    else:
        device = "cpu"
        print("Using CPU (training will be slow)")
    return device


def load_dataset_from_folder(data_dir: Path) -> DatasetDict:
    """
    Load dataset from ImageFolder structure.

    Expected structure:
        data_dir/
        ├── train/
        │   ├── ai/
        │   └── real/
        ├── val/
        │   ├── ai/
        │   └── real/
        └── test/
            ├── ai/
            └── real/
    """
    datasets = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping")
            continue

        images = []
        labels = []

        # Load real images (label = 0)
        real_dir = split_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    images.append(str(img_path))
                    labels.append(0)

        # Load AI images (label = 1)
        ai_dir = split_dir / "ai"
        if ai_dir.exists():
            # Direct files
            for img_path in ai_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    images.append(str(img_path))
                    labels.append(1)
            # Subdirectories (flux/, midjourney/, etc.)
            for subdir in ai_dir.iterdir():
                if subdir.is_dir():
                    for img_path in subdir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            images.append(str(img_path))
                            labels.append(1)

        if images:
            datasets[split] = Dataset.from_dict({
                "image_path": images,
                "label": labels,
            })
            print(f"{split}: {len(images)} images (Real: {labels.count(0)}, AI: {labels.count(1)})")

    if not datasets:
        raise ValueError(f"No valid data found in {data_dir}")

    return DatasetDict(datasets)


def create_transform_function(image_processor):
    """Create a transform function for the dataset."""
    def transform(example):
        try:
            img = Image.open(example["image_path"]).convert("RGB")
            inputs = image_processor(img, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "label": example["label"],
            }
        except Exception as e:
            print(f"Error loading {example['image_path']}: {e}")
            return None
    return transform


def collate_fn(batch):
    """Collate batch of examples, filtering out None values."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"],
    }


def train(
    data_dir: str,
    output_dir: str = "./checkpoints",
    base_model: str = "dima806/ai_vs_real_image_detection",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    push_to_hub: bool = False,
    hf_repo: str = None,
):
    """Run the training pipeline."""

    print("=" * 60)
    print("CapCheck AI Detection - Training")
    print("=" * 60)

    device = get_device()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Labels
    id2label = {0: "Real", 1: "Fake"}
    label2id = {"Real": 0, "Fake": 1}

    # Load base model
    print(f"\nLoading base model: {base_model}")
    image_processor = AutoImageProcessor.from_pretrained(base_model)
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    print(f"Model parameters: {model.num_parameters():,}")

    # Load dataset
    print(f"\nLoading dataset from: {data_dir}")
    dataset = load_dataset_from_folder(data_dir)

    if "train" not in dataset:
        raise ValueError("Training set (train/) is required")

    # Apply transforms
    transform = create_transform_function(image_processor)
    for split in dataset:
        dataset[split].set_transform(transform)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=device == "cuda",
        gradient_accumulation_steps=2,
        eval_strategy="epoch" if "val" in dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end="val" in dataset,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("val"),
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Train
    print("\n" + "-" * 60)
    print("Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print("-" * 60 + "\n")

    trainer.train()

    # Save final model
    final_path = output_dir / "capcheck-ai-detection-final"
    trainer.save_model(str(final_path))
    image_processor.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")

    # Evaluate on test set
    if "test" in dataset:
        print("\n" + "-" * 60)
        print("Evaluating on test set...")
        print("-" * 60)

        test_results = trainer.evaluate(dataset["test"])
        for key, value in test_results.items():
            if key.startswith("eval_"):
                print(f"  {key.replace('eval_', ''):15}: {value:.4f}")

        # Detailed classification report
        predictions = trainer.predict(dataset["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=["Real", "AI"]))

        print("Confusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print(f"  Real predicted as Real: {cm[0][0]}")
        print(f"  Real predicted as AI:   {cm[0][1]} (false positives)")
        print(f"  AI predicted as Real:   {cm[1][0]} (false negatives)")
        print(f"  AI predicted as AI:     {cm[1][1]}")

    # Push to HuggingFace Hub
    if push_to_hub:
        if not hf_repo:
            print("\nWarning: --hf-repo not specified, skipping push to hub")
        else:
            print(f"\nPushing to HuggingFace Hub: {hf_repo}")
            trainer.push_to_hub(repo_id=hf_repo, commit_message="Fine-tuned model")
            image_processor.push_to_hub(hf_repo)
            print(f"Published to: https://huggingface.co/{hf_repo}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {final_path}")
    print("\nNext steps:")
    print("1. Review the evaluation metrics above")
    print("2. If satisfied, publish to HuggingFace:")
    print(f"   python publish_to_hf.py {final_path}")
    print("3. Update predict.py MODEL_REGISTRY")
    print("4. Push to Replicate")

    return str(final_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT for AI image detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_vit.py --data-dir ./data

    # Custom hyperparameters
    python train_vit.py --data-dir ./data --epochs 5 --batch-size 8 --lr 1e-5

    # Train and push to HuggingFace
    python train_vit.py --data-dir ./data --push-to-hub --hf-repo capcheck/ai-image-detection
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory with train/val/test subdirs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="dima806/ai_vs_real_image_detection",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repository name (e.g., capcheck/ai-image-detection)",
    )

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        push_to_hub=args.push_to_hub,
        hf_repo=args.hf_repo,
    )


if __name__ == "__main__":
    main()
