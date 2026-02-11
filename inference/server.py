"""
Flask server wrapping the AI Detection Predictor
Compatible with Cog API format for drop-in replacement
"""

import os
import sys
import base64
import tempfile
import traceback
from flask import Flask, request, jsonify
from predict import Predictor

app = Flask(__name__)

# Initialize predictor on startup
predictor = None


def log(msg):
    """Print with flush for Docker logs"""
    print(msg, file=sys.stderr, flush=True)


def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
        predictor.setup()
    return predictor


@app.route("/health-check", methods=["GET"])
def health_check():
    """Health check endpoint for Fly.io"""
    return jsonify({"status": "healthy"})


@app.route("/predictions", methods=["POST"])
def predictions():
    """
    Cog-compatible predictions endpoint
    Expects: {"input": {"image": "data:image/...;base64,...", "threshold": 0.5}}
    Returns: {"output": {...}, "status": "succeeded"}
    """
    tmp_path = None
    try:
        log("📥 Received prediction request")
        data = request.get_json()

        if not data or "input" not in data:
            log("❌ Missing 'input' in request body")
            return jsonify({
                "output": None,
                "status": "failed",
                "error": "Missing 'input' in request body"
            }), 400

        input_data = data["input"]

        # Extract base64 image
        image_data = input_data.get("image", "")
        threshold = input_data.get("threshold", 0.8)
        log(f"   Threshold: {threshold}, Image data length: {len(image_data)} chars")

        # Handle data URL format
        if image_data.startswith("data:"):
            # Extract base64 part after comma
            image_data = image_data.split(",", 1)[1]
            log(f"   Extracted base64: {len(image_data)} chars")

        # Decode base64 to temp file
        log("   Decoding base64...")
        image_bytes = base64.b64decode(image_data)
        log(f"   Decoded to {len(image_bytes)} bytes")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        log(f"   Saved to temp file: {tmp_path}")

        try:
            # Run prediction
            log("   Running prediction...")
            pred = get_predictor()
            result = pred.predict(image=tmp_path, threshold=threshold)
            log(f"✅ Prediction complete: is_ai={result.get('is_ai_generated')}, confidence={result.get('confidence')}")

            return jsonify({
                "output": result,
                "status": "succeeded"
            })
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        log(f"❌ Prediction error: {e}")
        log(traceback.format_exc())
        return jsonify({
            "output": None,
            "status": "failed",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    # Pre-load model on startup
    print("Loading model...")
    get_predictor()
    print("Model loaded, starting server...")

    # Run with gunicorn in production, Flask dev server locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
