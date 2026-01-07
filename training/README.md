# Training Pipeline

Fine-tune the ViT model for AI image detection.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Prepare your data in the following structure:
# data/
#   train/
#     real/
#     ai/
#   val/
#     real/
#     ai/

python train_vit.py \
  --data_dir ./data \
  --output_dir ./checkpoints \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 2e-5
```

## Publishing

After training, publish to HuggingFace:

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token

# Publish the model
python publish_to_hf.py ./checkpoints/best-model

# Or update the base model
python publish_base_model.py
```

## Jupyter Notebook

For interactive training, see `capcheck_vit_training.ipynb`.
