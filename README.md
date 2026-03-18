# Prompted Segmentation for Drywall QA

Text-prompted segmentation using CLIPSeg to detect cracks and drywall taping areas. One model, two tasks, just change the prompt.

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Prepare data (generates masks and splits)
python src/prepare_data.py \
    --cracks_dir /path/to/cracks \
    --drywall_dir /path/to/Drywall-Join-Detect

# Train (freezes CLIP encoders, only trains decoder)
python src/train_clipseg.py \
    --data_dir ./data \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --epochs 20 \
    --device cuda  # or mps/cpu

# Evaluate
python src/evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data \
    --output_dir ./evaluation_results

# Visualize
python src/visualize.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data \
    --output_dir ./results \
    --cracks_threshold 0.42 \
    --drywall_threshold 0.36
```

## What This Does

You give it an image and a text prompt like `"segment crack"` or `"segment taping area"`, and it returns a binary mask showing where those things are. No separate models, no retraining for new defect types.

**The approach:** Freeze CLIP's vision and text encoders (they already know how to connect language and images), then train only a lightweight decoder to turn those connections into segmentation masks. This keeps training fast and prevents overfitting.

## Dataset

- **Cracks:** 5,369 images with polygon annotations → 3,758 train / 805 val / 806 test
- **Drywall:** 1,022 images with bbox annotations → 818 train / 101 val / 101 test
- **Combined:** 6,391 images total

The 5:1 imbalance is handled by concatenating both datasets and shuffling during training, so every batch has a mix of both tasks.

## Results

| Task | mIoU | Dice | Dice Std | F1 | Threshold | Samples |
|------|------|------|----------|----|-----------|---------| 
| Cracks | 0.430 | 0.568 | ±0.224 | 0.568 | 0.42 | 806 |
| Drywall | 0.573 | 0.718 | ±0.125 | 0.718 | 0.36 | 101 |
| **Average** | **0.501** | **0.643** | - | **0.643** | - | 907 |

Drywall is easier (bigger, more uniform regions), cracks are harder (thin, irregular). The model handles both reasonably well with task-specific thresholds.

See [REPORT](Report-prompted-segmentation.pdf) for the full writeup and analysis.

## Technical Details

**Model:** CLIPSeg (CIDAS/clipseg-rd64-refined)
- 150.7M total parameters, only 1.1M trainable (0.75%)
- CLIP encoders frozen, decoder trained from scratch
- Prompts: `"segment crack"` and `"segment taping area"`

**Training:**
- Loss: 50/50 mix of BCE and Dice
- Optimizer: AdamW (lr=1e-4, cosine annealing)
- Early stopping with patience=5
- Augmentations: flips, rotation, brightness/contrast, noise
- Time: ~2.5-3 hours on GTX 1050 Ti (4GB VRAM)

**Inference:**
- ~0.15-0.20 sec/image on GTX 1050 Ti
- Threshold optimization via grid search on validation set

## Project Structure

```
prompted-segmentation/
├── src/
│   ├── prepare_data.py     # Parse COCO, generate masks, create splits
│   ├── dataset.py          # PyTorch Dataset with augmentations
│   ├── train_clipseg.py    # Training loop with frozen encoders
│   ├── evaluate.py         # Metrics + threshold optimization
│   └── visualize.py        # Generate triptych visualizations
├── data/
│   ├── processed/          # Binary masks (generated)
│   └── splits/             # Train/val/test JSONs (generated)
├── checkpoints/            # Saved models
├── evaluation_results/     # Metrics and prediction masks
└── requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Common Issues

**Out of memory:** Reduce `--batch_size` to 4 or 2

**Missing split files:** Run `prepare_data.py` first

**Slow on CPU:** Use `--device cuda` (NVIDIA) or `--device mps` (Mac M-series)

## Limitations

- Only tested with one prompt per task (robustness to paraphrasing unknown)
- Drywall annotations are bboxes converted to rectangular masks (not precise boundaries)
- No baseline comparison with traditional segmentation models
- Generalization to new sites/lighting/cameras not validated

See [REPORT.md](REPORT.md) for detailed discussion.
