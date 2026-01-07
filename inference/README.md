# CapCheck AI Image Detection

Detect AI-generated images using a Vision Transformer (ViT) model. Returns confidence scores for real vs AI-generated classification.

## Use Cases

- **Fact-checking**: Verify if images in news articles are AI-generated
- **Content moderation**: Flag synthetic media before publication
- **Research**: Analyze datasets for AI-generated content

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

## Model Info

| Property | Value |
|----------|-------|
| Model | [capcheck/ai-image-detection](https://huggingface.co/capcheck/ai-image-detection) |
| Architecture | ViT-Base (86M parameters) |
| Input | 224x224 pixels |
| Output | Real/Fake classification |
| License | Apache 2.0 |

## Model Lineage

This model builds on excellent open-source work:

1. **Google** - [ViT-Base](https://huggingface.co/google/vit-base-patch16-224-in21k) architecture (Apache 2.0)
2. **dima806** - [Fine-tuned on CIFAKE dataset](https://huggingface.co/dima806/ai_vs_real_image_detection) for AI detection
3. **CapCheck** - Published with ongoing improvements for modern AI generators

---

## Development

### Local Development

```bash
# Build the container
cog build -t capcheck-ai-detection

# Test with an image
cog predict -i image=@path/to/test.jpg

# Run server locally
cog run -p 5000 python -m cog.server.http
```

## Deployment

### Deploy to Fly.io

```bash
cd ml/services/ai-image-detection
fly deploy -a capcheck-ai-image-detection
```

### Deploy to Replicate

```bash
cd ml/services/ai-image-detection
cog push r8.im/capcheck/ai-image-detection
```

## Updating the Model

When you fine-tune or update the model:

### Step 1: Publish to HuggingFace (Source of Truth)

```bash
cd ml/training

# For a new fine-tuned model:
python publish_to_hf.py ./checkpoints/your-new-model

# Or update the base model:
python publish_base_model.py
```

### Step 2: Redeploy Services

Services pull from HuggingFace at build time, so just redeploy:

```bash
cd ml/services/ai-image-detection

# Fly.io
fly deploy -a capcheck-ai-image-detection

# Replicate
cog push r8.im/capcheck/ai-image-detection
```

**That's it!** The new model weights are fetched from HuggingFace during the build.

### Model Versioning

Models are managed via `MODEL_REGISTRY` in `predict.py`:

```python
MODEL_REGISTRY = {
    "v1.0.0": "capcheck/ai-image-detection",  # Current
    # "v2.0.0": "capcheck/ai-image-detection-v2",  # Future
}
```

Set version via environment variable: `MODEL_VERSION=v1.0.0`

## Costs

- **Fly.io CPU**: ~$15/month (always-on)
- **Per-image**: ~$0.00005
- **Inference time**: 25-100ms on CPU

See `FINE_TUNING_PLAN.md` for full architecture documentation.

## License

- Model: Apache 2.0 (see [HuggingFace model card](https://huggingface.co/capcheck/ai-image-detection))
- Service code: CapCheck
