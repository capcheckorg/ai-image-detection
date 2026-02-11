#!/usr/bin/env python3
"""
Publish the human-generated image detection model to HuggingFace Hub.

This script downloads dima806's newer fine-tuned model and re-publishes it
under the CapCheck namespace with proper attribution.

Model Lineage:
- Base: google/vit-base-patch16-224-in21k (Google's ViT, Apache 2.0)
- Fine-tuned: dima806/ai_vs_human_generated_image_detection (2024)
- Re-published: capcheck/ai-human-generated-image-detection

This model uses different training data and labels than the original:
- Original (CIFAKE): labels "REAL" / "FAKE"
- This model: labels "human" / "AI-generated"

Usage:
    HF_TOKEN=hf_xxx python publish_human_generated_model.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModelForImageClassification, AutoImageProcessor


# Configuration
SOURCE_MODEL = "dima806/ai_vs_human_generated_image_detection"
TARGET_REPO = "capcheck/ai-human-generated-image-detection"
LOCAL_DIR = Path("./checkpoints/capcheck-human-generated")

MODEL_CARD = """---
license: apache-2.0
base_model: dima806/ai_vs_human_generated_image_detection
tags:
- image-classification
- vision
- ai-detection
- deepfake-detection
- vit
metrics:
- accuracy
- f1
pipeline_tag: image-classification
---

# CapCheck AI vs Human-Generated Image Detection

Vision Transformer (ViT) fine-tuned for detecting AI-generated images.
Uses newer training data than the original CIFAKE-based model.

## Model Lineage & Attribution

This model builds on the work of others:

| Layer | Model | Author | License |
|-------|-------|--------|---------|
| Base Architecture | [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) | Google | Apache 2.0 |
| AI Detection Fine-tune | [dima806/ai_vs_human_generated_image_detection](https://huggingface.co/dima806/ai_vs_human_generated_image_detection) | dima806 | Apache 2.0 |
| This Model | capcheck/ai-human-generated-image-detection | CapCheck | Apache 2.0 |

**Special thanks to:**
- **Google** for the Vision Transformer (ViT) architecture
- **dima806** for fine-tuning on AI vs human-generated image datasets

## Model Description

- **Architecture**: ViT-Base (86M parameters)
- **Input Size**: 224x224 pixels
- **Training Data**: AI-generated vs human-created images (newer dataset than CIFAKE)
- **Task**: Binary classification (human vs AI-generated)

## Usage

```python
from transformers import pipeline

detector = pipeline("image-classification", model="capcheck/ai-human-generated-image-detection")
result = detector("path/to/image.jpg")

# Output:
# [{"label": "AI-generated", "score": 0.95}, {"label": "human", "score": 0.05}]
```

## Labels

| Label | Description |
|-------|-------------|
| `human` | Authentic photograph or human-created image |
| `AI-generated` | AI-generated or synthetically created image |

## Comparison with capcheck/ai-image-detection

| Feature | ai-image-detection | ai-human-generated-image-detection |
|---------|-------------------|-------------------------------------|
| Source | dima806/ai_vs_real_image_detection | dima806/ai_vs_human_generated_image_detection |
| Training Data | CIFAKE dataset | Newer AI vs human dataset |
| Labels | `REAL` / `FAKE` | `human` / `AI-generated` |
| Handle | `ai-image-detection` | `ai-human-generated-image-detection` |

## Limitations

- May have reduced accuracy on very new AI generators not in training data
- Heavily compressed images (low JPEG quality) can affect results
- Works best on images with clear subjects (224x224+ pixels)

## Intended Use

- Content moderation and authenticity verification
- Research into AI-generated content detection
- Educational purposes

**Not intended for**:
- Making consequential decisions without human review
- Law enforcement evidence without corroborating sources

## License

Apache 2.0 (inherited from Google ViT and dima806's fine-tuned model)

## Citation

```bibtex
@misc{capcheck-ai-human-generated-detection,
  author = {CapCheck},
  title = {AI vs Human-Generated Image Detection Model},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/capcheck/ai-human-generated-image-detection},
  note = {Based on dima806/ai_vs_human_generated_image_detection, fine-tuned from google/vit-base-patch16-224-in21k}
}
```
"""


def main():
    """Download and publish the model to HuggingFace."""

    # Check for HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"Logging into HuggingFace...")
    login(token=token)

    # Download source model
    print(f"\nDownloading model from: {SOURCE_MODEL}")
    model = AutoModelForImageClassification.from_pretrained(SOURCE_MODEL)
    processor = AutoImageProcessor.from_pretrained(SOURCE_MODEL)

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Model labels: {model.config.id2label}")

    # Save locally
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {LOCAL_DIR}")
    model.save_pretrained(LOCAL_DIR)
    processor.save_pretrained(LOCAL_DIR)

    # Write model card
    readme_path = LOCAL_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)
    print(f"Model card written to: {readme_path}")

    # Create repo and push
    api = HfApi()

    print(f"\nCreating/updating repository: {TARGET_REPO}")
    try:
        api.create_repo(
            repo_id=TARGET_REPO,
            repo_type="model",
            exist_ok=True,
        )
    except Exception as e:
        print(f"Note: {e}")

    print(f"Pushing model to HuggingFace Hub...")
    model.push_to_hub(TARGET_REPO, commit_message="Upload AI vs human-generated detection model")
    processor.push_to_hub(TARGET_REPO, commit_message="Upload image processor")

    # Upload README
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=TARGET_REPO,
        commit_message="Add model card with attribution",
    )

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nModel published to: https://huggingface.co/{TARGET_REPO}")
    print("\nNext steps:")
    print("1. Update inference/predict.py MODEL_REGISTRY (already done if following the plan)")
    print("2. Rebuild Docker: docker compose build ai-image-detection")
    print("3. Deploy to Fly.io: fly deploy -a capcheck-ai-image-detection")


if __name__ == "__main__":
    main()
