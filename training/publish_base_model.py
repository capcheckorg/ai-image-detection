#!/usr/bin/env python3
"""
Publish the AI image detection model to HuggingFace Hub.

This script downloads dima806's fine-tuned model and re-publishes it
under the CapCheck namespace with proper attribution.

Model Lineage:
- Base: google/vit-base-patch16-224-in21k (Google's ViT, Apache 2.0)
- Fine-tuned: dima806/ai_vs_real_image_detection (CIFAKE dataset, 2024)
- Re-published: capcheck/ai-image-detection

Usage:
    HF_TOKEN=hf_xxx python publish_base_model.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModelForImageClassification, AutoImageProcessor


# Configuration
SOURCE_MODEL = "dima806/ai_vs_real_image_detection"
TARGET_REPO = "capcheck/ai-image-detection"
LOCAL_DIR = Path("./checkpoints/capcheck-base")

MODEL_CARD = """---
license: apache-2.0
base_model: dima806/ai_vs_real_image_detection
tags:
- image-classification
- vision
- ai-detection
- deepfake-detection
- vit
datasets:
- CIFAKE
metrics:
- accuracy
- f1
pipeline_tag: image-classification
---

# CapCheck AI Image Detection

Vision Transformer (ViT) fine-tuned for detecting AI-generated images.

## Model Lineage & Attribution

This model builds on the work of others:

| Layer | Model | Author | License |
|-------|-------|--------|---------|
| Base Architecture | [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) | Google | Apache 2.0 |
| AI Detection Fine-tune | [dima806/ai_vs_real_image_detection](https://huggingface.co/dima806/ai_vs_real_image_detection) | dima806 | Apache 2.0 |
| This Model | capcheck/ai-image-detection | CapCheck | Apache 2.0 |

**Special thanks to:**
- **Google** for the Vision Transformer (ViT) architecture
- **dima806** for fine-tuning on the CIFAKE dataset for AI image detection

## Model Description

- **Architecture**: ViT-Base (86M parameters)
- **Input Size**: 224x224 pixels
- **Training Data**: CIFAKE dataset (AI-generated vs real images)
- **Task**: Binary classification (Real vs Fake/AI-generated)

## Usage

```python
from transformers import pipeline

detector = pipeline("image-classification", model="capcheck/ai-image-detection")
result = detector("path/to/image.jpg")

# Output:
# [{"label": "Fake", "score": 0.95}, {"label": "Real", "score": 0.05}]
```

## Labels

| Label | Description |
|-------|-------------|
| `Real` | Authentic photograph or real-world image |
| `Fake` | AI-generated or synthetically created image |

## Performance

This model was trained on the CIFAKE dataset. Performance on modern AI generators
(Flux, Midjourney v6, DALL-E 3, Stable Diffusion 3) may vary.

See [dima806's model card](https://huggingface.co/dima806/ai_vs_real_image_detection)
for detailed training metrics.

## Limitations

- Trained primarily on older AI generators (pre-2024)
- May have reduced accuracy on:
  - Very new AI generators not in training data
  - Heavily compressed images (low JPEG quality)
  - Images smaller than 224x224 pixels
- Works best on images with clear subjects

## Intended Use

- Content moderation and authenticity verification
- Research into AI-generated content detection
- Educational purposes

**Not intended for**:
- Making consequential decisions without human review
- Law enforcement evidence without corroborating sources

## Ethical Considerations

- This tool is not 100% accurate - false positives harm legitimate creators
- False negatives can allow misinformation to spread
- Use in conjunction with other verification methods
- Human review is recommended for high-stakes decisions

## Roadmap

### Current Version (v1.0.0)

Base model from dima806's CIFAKE-trained ViT. Solid foundation for AI detection.

### Planned Improvements

**Phase 1: Modern Generator Training**
- Fine-tune on images from Flux, Midjourney v6, DALL-E 3, Stable Diffusion 3
- Target: Reduce false negatives on 2024+ AI generators

**Phase 2: False Positive Reduction**
- Curate dataset of real images commonly flagged as AI
- Photography edge cases: HDR, heavy editing, digital art
- Target: <5% false positive rate

**Phase 3: Continuous Updates**
- Quarterly re-training as new generators emerge
- Community feedback integration
- Benchmark against latest AI generators

### Contributing

We welcome:
- Dataset contributions (properly licensed images)
- Bug reports and false positive/negative examples
- Benchmark results on new generators

Join the discussion: https://huggingface.co/capcheck/ai-image-detection/discussions

## License

Apache 2.0 (inherited from Google ViT and dima806's fine-tuned model)

## Citation

If you use this model, please cite:

```bibtex
@misc{capcheck-ai-detection,
  author = {CapCheck},
  title = {AI Image Detection Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/capcheck/ai-image-detection},
  note = {Based on dima806/ai_vs_real_image_detection, fine-tuned from google/vit-base-patch16-224-in21k}
}
```

## Changelog

### v1.0.0 (Initial Release)

- Published base model from dima806/ai_vs_real_image_detection
- Added proper attribution and documentation
- Established as CapCheck's source of truth for AI image detection
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
    model.push_to_hub(TARGET_REPO, commit_message="Upload AI detection model")
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
    print("1. Update ml/services/ai-image-detection/predict.py MODEL_REGISTRY")
    print("2. Deploy to Fly.io: fly deploy -a capcheck-ai-image-detection")
    print("3. Deploy to Replicate: cog push r8.im/capcheck/ai-image-detection")


if __name__ == "__main__":
    main()
