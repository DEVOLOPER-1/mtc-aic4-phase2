# AbaViTrack — Object Tracking Inference

**Team:** Zerone
**Affiliation:** University of Science and Technology, Zewail City

---

## Overview

This repository implements an object tracker based on the **AbaViTrack** architecture — a Vision Transformer (ViT) tracker with Adaptive Synergy, Hanning-window response penalisation, cosine-similarity feature memory, and dynamic search-factor expansion for out-of-view (OOV) recovery.

The tracker:

- Takes a video sequence as input
- Predicts one bounding box per frame
- Outputs predictions in CSV format compatible with the competition evaluation script

---

## Architecture

| Component | Details |
|-----------|---------|
| Backbone | AbaViT (`abavit_patch16_224`) — patch size 16, hidden dim 192 |
| Head | CENTER box head with 256 channels |
| Template size | 128 × 128 |
| Search size | 256 × 256 |
| Backbone stride | 16 |
| CE locations | Layers [3, 6, 9] |
| Key innovations | Adaptive Synergy, cosine similarity memory, dynamic search expansion |

---

## Checkpoints

Two model checkpoints are supported:

| Checkpoint | Path | Description                                               |
|------------|------|-----------------------------------------------------------|
| Full-precision | `checkpoints/model.pth` | Original FP32 checkpoint (download via `download.py`) or https://drive.google.com/drive/folders/1Ii08izcdhpvRP-O97IKXhrzzwyTvNUgY?usp=sharing |
| Quantized | `checkpoints/model_quantized.pth` | https://drive.google.com/drive/folders/1M4pKiVxIzu4DK3d9ENDoiNMykyhWPgcN?usp=sharing               |

> **Note:** The quantized checkpoint is produced by post-training quantization for edge deployment.
---

## Prerequisites

- Python ≥ 3.12
- CUDA-capable GPU (optional — falls back to CPU)
- `uv` package manager (recommended) or `pip`

---

## Installation

### Option A — Using `uv` (recommended)

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Option B — Using `pip` directly

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.12.1 | Model inference |
| `opencv-python` | ≥ 4.10.0 | Video I/O, image processing |
| `pandas` | ≥ 2.2.0 | CSV validation |
| `scipy` | ≥ 1.14.0 | Signal processing |
| `timm` | ≥ 1.0.0 | Vision transformer utilities |
| `gdown` | ≥ 6.1.0 | Google Drive checkpoint download |

---

## Step 1 — Download Checkpoints

```bash
python download.py
```

This downloads the full-precision checkpoint into `checkpoints/model.pth`.


For the quantized checkpoint, place it at:
```
checkpoints/model_quantized.pth
```

---

## Step 2 — Run Inference

### Full-precision checkpoint (default)

```bash
python inference.py \
    data/test.json \
    hidden \
    predictions.csv
```

### Quantized checkpoint

To use the quantized checkpoint, modify the `CHECKPOINT_PATH` in `predictor.py`:

```python
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "model_quantized.pth"
```

Then run:

```bash
python inference.py \
    data/test.json \
    hidden \
    predictions_quantized.csv
```

### Arguments

| Argument | Description |
|----------|-------------|
| `data/test.json` | Path to dataset JSON (contains video paths and annotations) |
| `hidden` | Split name inside the JSON (e.g., `test`, `hidden`, `val`) |
| `predictions.csv` | Output CSV file path |

### CPU-only inference

The script auto-detects CUDA. To force CPU:

```python
# In predictor.py, change DEVICE line or set:
DEVICE = "cpu"
```

Or modify `inference.py` line 24–28 to default to `"cpu"`.

---

## Step 3 — Validate Submission

```bash
python check_submission.py \
    sample_submission.csv \
    predictions.csv
```

This verifies:
- Correct CSV columns (`id, x, y, w, h`)
- Correct frame IDs matching the ground truth
- Correct number of predictions (one per frame per sequence)

---

## End-to-End Verification

Full pipeline from scratch:

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Download checkpoint
python download.py

# 3. Run inference (full-precision)
python inference.py data/test.json hidden predictions.csv

# 4. Validate submission format
python check_submission.py sample_submission.csv predictions.csv

# 5. (Optional) Run inference (quantized)
#    → update CHECKPOINT_PATH in predictor.py to model_quantized.pth
python inference.py data/test.json hidden predictions_quantized.csv
python check_submission.py sample_submission.csv predictions_quantized.csv
```

---

## Output Format

### CSV Format

```csv
id,x,y,w,h
dataset1/Car_video_0,10.5,20.3,30.1,40.2
dataset1/Car_video_1,11.0,21.0,30.5,40.8
```

### Prediction Structure

Each prediction is a dict:

```python
{
    "frame_idx": 0,
    "x": 10.0,
    "y": 20.0,
    "w": 30.0,
    "h": 40.0,
}
```

---

## Project Structure

```
.
├── predictor.py          # Tracker model (load_model, run_tracker)
├── inference.py          # Inference entry point (DO NOT MODIFY)
├── download.py           # Checkpoint download helper
├── check_submission.py   # Submission validator (DO NOT MODIFY)
├── requirements.txt      # Pinned dependencies
├── pyproject.toml        # Project metadata (uv/pip)
├── checkpoints/          # Model weights directory
│   ├── model.pth         # Full-precision checkpoint
│   └── model_quantized.pth  # [PLACEHOLDER — quantized checkpoint]
├── data/                 # Dataset directory
│   └── test.json         # Test split manifest
└── sample_submission.csv # Ground-truth format reference
```

---

## Key Implementation Details

### Adaptive Synergy

The tracker maintains a cosine-similarity memory of the target's feature representation. When confidence drops below a dynamic threshold, the tracker enters a "PANIC STATE" — freezing the bounding box and expanding the search window up to 16× to recover the target.

### Hanning Window

A 2-D Hanning window is applied to the response map to penalise boundary effects and encourage centre-weighted predictions.

### Dynamic Search Factor

The search region scales adaptively:
- **FOCUS STATE** (target confirmed): search factor = 4.5×
- **PANIC STATE** (OOV/distractor): search factor × 1.5 per frame, up to 16×

### cuDNN Handling

cuDNN is disabled during model loading for stable GPU inference on embedded platforms (e.g., Jetson Orin):

```python
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
```

---

## References

| Resource | Link/Location |
|----------|---------------|
| Original Paper | https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Adaptive_and_Background-Aware_Vision_Transformer_for_Real-Time_UAV_Tracking_ICCV_2023_paper.pdf |
| Optimization Report | https://drive.google.com/drive/folders/1KQfS5pRW-1VqE3slSg72Ztb_hQpycxtN?usp=sharing|

---

## Rules Compliance

- ✅ PyTorch only
- ✅ OpenCV for video I/O
- ✅ No absolute paths
- ✅ No interactive input
- ✅ No modification to `inference.py`
- ✅ Relative paths only
- ✅ No internet access required at inference time

---

## Team

**Zerone** — University of Science and Technology, Zewail City
