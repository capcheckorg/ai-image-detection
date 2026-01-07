"""
CapCheck AI Detection Service

Detects AI-generated images using a Vision Transformer (ViT) model.
Supports model versioning for easy updates and fine-tuning.

Model Lineage:
- Base: google/vit-base-patch16-224-in21k (Google's ViT)
- Fine-tuned: dima806/ai_vs_real_image_detection (CIFAKE dataset, 2024)
- Published: capcheck/ai-image-detection (HuggingFace)

Version: 1.0.0
License: Apache 2.0
"""

from pathlib import Path as PathLib
try:
    from cog import Path  # Replicate
except ImportError:
    from pathlib import Path  # Fly.io
from transformers import pipeline
from PIL import Image
import os
import sys
import time
import gc

# Model versioning - easily swap models via environment variable
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")

MODEL_REGISTRY = {
    "v1.0.0": "capcheck/ai-image-detection",  # Our published model (HuggingFace source of truth)
    # "v2.0.0": "capcheck/ai-image-detection",  # Future fine-tuned version
}


class Predictor:
    def setup(self):
        """Load model into GPU memory on startup"""
        model_name = MODEL_REGISTRY.get(MODEL_VERSION, MODEL_REGISTRY["v1.0.0"])
        self.model_version = MODEL_VERSION

        print(f"Loading model: {model_name} (version: {MODEL_VERSION})")
        start_time = time.time()

        # Force CPU inference - ViT-Base (86M params) runs efficiently on CPU
        # ~25-100ms inference time, 10x cheaper than GPU
        self.pipe = pipeline(
            "image-classification",
            model=model_name,
            device=-1  # CPU only
        )

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Device: CPU (forced for cost efficiency)")

    def predict(self, image: Path, threshold: float = 0.5) -> dict:
        """
        Detect if an image is AI-generated.

        Returns:
            dict with detection results including confidence scores
        """
        start_time = time.time()
        img = None

        try:
            # Load and preprocess image
            print(f"   Loading image: {image}", file=sys.stderr, flush=True)
            img = Image.open(image).convert("RGB")
            original_size = img.size
            print(f"   Original size: {original_size}", file=sys.stderr, flush=True)

            # Aggressive resize - ViT model uses 224x224 internally
            # Keeping at 512 max to reduce memory while preserving enough detail
            max_size = 512
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"   Resized to: {img.size}", file=sys.stderr, flush=True)

            # Run inference
            print("   Running ViT inference...", file=sys.stderr, flush=True)
            results = self.pipe(img)
            print(f"   Inference complete, results: {results}", file=sys.stderr, flush=True)

            # Parse results - model returns labels like "Fake" and "Real"
            ai_score = 0.0
            real_score = 0.0

            for r in results:
                label = r["label"].lower()
                score = r["score"]

                # Handle various label formats from different model versions
                if any(term in label for term in ["fake", "ai", "generated", "synthetic", "deepfake"]):
                    ai_score = score
                elif any(term in label for term in ["real", "human", "authentic", "genuine"]):
                    real_score = score

            # Determine if AI-generated based on threshold
            is_ai_generated = ai_score > threshold

            # Calculate inference time
            inference_time_ms = (time.time() - start_time) * 1000

            return {
                "is_ai_generated": is_ai_generated,
                "confidence": ai_score if is_ai_generated else real_score,
                "ai_probability": round(ai_score, 4),
                "real_probability": round(real_score, 4),
                "threshold": threshold,
                "model_version": self.model_version,
                "inference_time_ms": round(inference_time_ms, 2),
                "raw_results": results
            }

        except Exception as e:
            print(f"   ❌ Prediction error: {e}", file=sys.stderr, flush=True)
            raise

        finally:
            # Explicit cleanup to prevent memory leaks
            if img is not None:
                del img
            gc.collect()
