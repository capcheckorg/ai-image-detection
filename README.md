# AI Image Detection

Detect AI-generated images using a Vision Transformer (ViT) model. Open source inference server and training pipeline.

[![Docker](https://github.com/capcheckorg/ai-image-detection/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/capcheckorg/ai-image-detection/actions/workflows/docker-publish.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

- **Fast inference** - 25-100ms per image on CPU
- **High accuracy** - ViT-Base fine-tuned on AI vs real image datasets
- **Docker-ready** - Pre-built images on GHCR
- **Cog-compatible** - Deploy to Replicate or run locally

## Quick Start

### Using Docker

```bash
# Pull the pre-built image
docker pull ghcr.io/capcheckorg/ai-image-detection:latest

# Run the server
docker run -p 5000:5000 ghcr.io/capcheckorg/ai-image-detection:latest

# Test with an image
curl -X POST http://localhost:5000/predictions \
  -F "input=@path/to/image.jpg"
```

### Using Cog

```bash
cd inference

# Build
cog build -t ai-image-detection

# Run prediction
cog predict -i image=@path/to/image.jpg

# Start server
cog run -p 5000 python -m cog.server.http
```

## API

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file/URL | required | Image to analyze (JPEG, PNG, WebP) |
| `threshold` | float | 0.5 | Classification threshold (0.0-1.0) |

### Output

```json
{
  "is_ai_generated": true,
  "confidence": 0.97,
  "ai_probability": 0.97,
  "real_probability": 0.03,
  "threshold": 0.5,
  "model_version": "v1.0.0",
  "inference_time_ms": 45.2
}
```

## Model

| Property | Value |
|----------|-------|
| Architecture | ViT-Base (86M parameters) |
| Input | 224x224 pixels |
| HuggingFace | [capcheck/ai-image-detection](https://huggingface.co/capcheck/ai-image-detection) |
| License | Apache 2.0 |

### Model Lineage

This model builds on open-source work:

1. **Google** - [ViT-Base](https://huggingface.co/google/vit-base-patch16-224-in21k) architecture
2. **dima806** - [Fine-tuned on CIFAKE dataset](https://huggingface.co/dima806/ai_vs_real_image_detection)
3. **CapCheck** - Published with ongoing improvements for modern AI generators

## Project Structure

```
├── inference/          # Inference server
│   ├── predict.py      # Core prediction logic
│   ├── server.py       # HTTP server
│   ├── Dockerfile
│   ├── cog.yaml
│   └── requirements.txt
├── training/           # Training pipeline
│   ├── train_vit.py    # Fine-tuning script
│   ├── publish_to_hf.py
│   └── requirements.txt
└── .github/workflows/  # CI/CD
```

## Training

See [training/](training/) for fine-tuning instructions.

```bash
cd training
pip install -r requirements.txt

# Fine-tune on your dataset
python train_vit.py --data_dir ./data --output_dir ./checkpoints

# Publish to HuggingFace
python publish_to_hf.py ./checkpoints/best-model
```

## Use Cases

- **Fact-checking**: Verify if images in news articles are AI-generated
- **Content moderation**: Flag synthetic media before publication
- **Research**: Analyze datasets for AI-generated content

## License

Apache 2.0 - See [LICENSE](LICENSE)
