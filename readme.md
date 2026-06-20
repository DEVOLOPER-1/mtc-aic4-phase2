# Object Tracking Competition

## Overview

Participants must implement an object tracker that:

- takes a video sequence as input
- predicts one bounding box per frame
- outputs predictions in CSV format

The organizer evaluation script automatically:

- loads videos
- measures latency
- saves predictions
- validates submission format

Participants only modify:

- `predictor.py`

---

# Repository Structure

```text
submission/
├── inference.py
├── predictor.py
├── download.py
├── check_submission.py
├── requirements.txt
├── sample_submission.csv
└── checkpoints/
```

---

# Installation

Create environment:
 requirements should include the exact versions not just the libraries
```bash
pip install -r requirements.txt
```

---

# Step 1 — Download Checkpoints

Download your model weights:

```bash
python download.py
```

This should download checkpoints into:

```text
checkpoints/
```

Example:

```text
checkpoints/model.pth
```
> You are required to replace "YOUR_FILE_ID" with the actual file ID from Google Drive to download the checkpoint.
Or replace the download method with your preferred method if you are not using Google Drive.
>
> If we cannot download the file successfully, your team will be disqualified 
---

# Step 2 — Run Inference

Run tracking inference:

```bash
python inference.py \
    data/test.json \
    split_name \
    predictions.csv
```

Arguments:

```text
1. input json
2. split name
3. output csv
```


---

# Step 3 — Validate Submission

Check that your submission format is correct:

```bash
python check_submission.py \
    sample_submission.csv \
    predictions.csv
```

This verifies:

- correct CSV columns
- correct frame IDs
- correct number of predictions

---

# Required CSV Format

Your predictions must follow:

```csv
id,x,y,w,h
dataset1/Car_video_0,0,0,0,0
dataset1/Car_video_1,0,0,0,0
dataset1/Car_video_2,0,0,0,0
```

---

# Required Output

Your tracker must return:

```python
[
    {
        "frame_idx": 0,
        "x": 10,
        "y": 20,
        "w": 30,
        "h": 40,
    }
]
```

One prediction per frame.

---

# Rules

## Allowed

- PyTorch
- OpenCV
- Any tracking architecture
- Any Python libraries in `requirements.txt`

## Not Allowed

- Absolute paths
- Interactive input
- Manual file selection
- Modifying `inference.py`

---

# Notes

- The first-frame bounding box is provided.
- One bounding box must be predicted for every frame.
- Relative paths only.
- The evaluation environment may not have internet access during inference.

---

# Another reminder this should  be what we will do (any failure in this will lead to immediate disqualification) 

```bash
# install dependencies
pip install -r requirements.txt

# download checkpoint
python download.py

# run inference
python inference.py \
    data/test.json \
    hidden \
    predictions.csv

# validate predictions
python check_submission.py \
    sample_submission.csv \
    predictions.csv
```