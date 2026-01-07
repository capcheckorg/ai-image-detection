#!/usr/bin/env python3
"""
Publish a trained CapCheck AI detection model to HuggingFace Hub.

Usage:
    python publish_to_hf.py ./checkpoints/capcheck-ai-detection-final

Environment:
    HF_TOKEN: HuggingFace API token (or use `huggingface-cli login`)
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModelForImageClassification, AutoImageProcessor


def create_model_card(
    model_name: str,
    base_model: str = "dima806/ai_vs_real_image_detection",
    training_info: dict = None,
) -> str:
    """Generate a model card for the HuggingFace Hub."""

    training_info = training_info or {}

    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
- image-classification
- vision
- ai-detection
- deepfake-detection
- vit
datasets:
- custom
metrics:
- accuracy
- f1
- precision
- recall
pipeline_tag: image-classification
---

# CapCheck AI Image Detection

Fine-tuned Vision Transformer (ViT) for detecting AI-generated images.

## Model Description

This model is fine-tuned from `{base_model}` on a custom dataset of modern AI-generated images including:
- Flux
- Midjourney v6
- DALL-E 3
- Stable Diffusion 3
- And more...

The goal is to accurately distinguish between real photographs and AI-generated images, with a focus on:
- Reducing false positives (real images incorrectly flagged as AI)
- Reducing false negatives (AI images slipping through as real)

## Usage

```python
from transformers import pipeline

detector = pipeline("image-classification", model="{model_name}")
result = detector("path/to/image.jpg")

# Result format:
# [
#   {{"label": "Fake", "score": 0.95}},
#   {{"label": "Real", "score": 0.05}}
# ]
```

## Labels

| Label | Description |
|-------|-------------|
| `Real` | Authentic photograph or real-world image |
| `Fake` | AI-generated or synthetically created image |

## Performance

{_format_metrics(training_info.get('metrics', {}))}

## Training Details

- **Base Model**: {base_model}
- **Architecture**: Vision Transformer (ViT-Base, 86M parameters)
- **Input Size**: 224x224 pixels

## Limitations

- Performance may vary on image types not seen during training
- Heavily compressed images (low JPEG quality) may reduce accuracy
- Very new AI generators not in training data may evade detection
- Works best on images at least 224x224 pixels

## Intended Use

This model is designed for:
- Content moderation and authenticity verification
- Research into AI-generated content detection
- Educational purposes about AI image generation

## Ethical Considerations

- This tool is not 100% accurate and should not be the sole basis for consequential decisions
- False positives can harm legitimate content creators
- False negatives can allow misinformation to spread
- Use in conjunction with other verification methods

## License

Apache 2.0

## Citation

If you use this model, please cite:

```bibtex
@misc{{capcheck-ai-detection,
  author = {{CapCheck}},
  title = {{AI Image Detection Model}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{model_name}}}
}}
```
"""


def _format_metrics(metrics: dict) -> str:
    """Format metrics for the model card."""
    if not metrics:
        return "*Metrics will be updated after evaluation.*"

    lines = ["| Metric | Score |", "|--------|-------|"]
    for name, value in metrics.items():
        lines.append(f"| {name.capitalize()} | {value:.4f} |")
    return "\n".join(lines)


def publish_model(
    model_path: str,
    repo_name: str = "capcheck/ai-image-detection",
    base_model: str = "dima806/ai_vs_real_image_detection",
    private: bool = False,
    create_repo: bool = True,
):
    """Publish a trained model to HuggingFace Hub."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"Publishing model from: {model_path}")
    print(f"Target repository: {repo_name}")

    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found, attempting interactive login...")
        login()

    api = HfApi()

    # Create repository if needed
    if create_repo:
        try:
            api.create_repo(
                repo_id=repo_name,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            print(f"Repository ready: {repo_name}")
        except Exception as e:
            print(f"Repository creation note: {e}")

    # Load and push model
    print("Loading model...")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    print("Pushing model to Hub...")
    model.push_to_hub(repo_name, commit_message="Upload fine-tuned model")
    processor.push_to_hub(repo_name, commit_message="Upload image processor")

    # Create and upload model card
    print("Creating model card...")
    model_card = create_model_card(repo_name, base_model)

    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        commit_message="Add model card",
    )

    print("\n" + "=" * 50)
    print("SUCCESS!")
    print("=" * 50)
    print(f"Model published to: https://huggingface.co/{repo_name}")
    print("\nNext steps:")
    print("1. Update ml/services/ai-image-detection/predict.py MODEL_REGISTRY")
    print("2. Update cog.yaml to pre-download the new model")
    print("3. Push to Replicate: cog push")

    return repo_name


def main():
    parser = argparse.ArgumentParser(
        description="Publish a trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="capcheck/ai-image-detection",
        help="HuggingFace repository name (default: capcheck/ai-image-detection)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="dima806/ai_vs_real_image_detection",
        help="Base model used for fine-tuning",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )

    args = parser.parse_args()

    publish_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        base_model=args.base_model,
        private=args.private,
    )


if __name__ == "__main__":
    main()
