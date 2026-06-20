"""
Predictor Module

This module provides functionality for loading a pre-trained model and running tracking on video sequences.

This module provides:
- Loading a pre-trained model
- Reading initial bounding boxes
- Running a tracker on video sequences
"""

import os
import cv2
import torch
import sys
from download import download_checkpoint
    
def load_model(device="cuda"):
    """
    Load model + checkpoint here.

    This function is called ONCE before evaluation.

    Example responsibilities:
        - download checkpoint
        - create model
        - load weights
        - move model to GPU
        - set eval mode

    Return:
        model
    """


    checkpoint_path = download_checkpoint(file_id="YOUR_FILE_ID")


    model = ...

    model.to(device)

    model.eval()

    return model

def read_init_box(path):
    """
    Read first-frame bounding box.

    Example annotation format:
        x,y,w,h

    Example:
        100,50,80,120
    """

    with open(path, "r") as f:

        line = f.readline().strip()

    x, y, w, h = map(float, line.split(","))

    return [x, y, w, h]

def run_tracker(
    model,
    video_path,
    init_box_path,
):
    """
    Run tracking on ONE video sequence.

    Inputs:
        model:
            Loaded model from load_model()

        video_path:
            Path to video file

        init_box_path:
            Path to first-frame annotation

    Output:
        List[dict]

    Required format:
        [
            {
                "frame_idx": 0,
                "x": 10,
                "y": 20,
                "w": 30,
                "h": 40,
            }
        ]
    """

    predictions = []

    init_box = read_init_box(init_box_path)

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------
    #
    # You may replace OpenCV with:
    #   - decord
    #   - torchvision
    #   - ffmpeg
    #   - pyav
    #
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    # --------------------------------------------------------
    # OPTIONAL TRACKER STATE
    # --------------------------------------------------------
    #
    # Many trackers maintain internal state:
    #   - template features
    #   - hidden states
    #   - memory banks
    #
    # You may store them here.
    #
    # Delete if not needed.
    #
    # --------------------------------------------------------

    tracker_state = None

    # --------------------------------------------------------
    # PROCESS VIDEO FRAME-BY-FRAME
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ====================================================
        # FIRST FRAME
        # ====================================================
        #
        # Usually:
        #   - initialize tracker
        #   - extract template
        #   - save reference features
        #
        # The first prediction is usually:
        #   init_box
        #
        # ====================================================

        if frame_idx == 0:

            # Example:
            #
            # tracker_state = model.initialize(
            #     frame,
            #     init_box,
            # )

            pred_box = init_box

        # ====================================================
        # REMAINING FRAMES
        # ====================================================
        #
        # Replace this section with:
        #   - model inference
        #   - tracking logic
        #   - bbox prediction
        #
        # Example:
        #
        # pred_box, tracker_state = model.track(
        #     frame,
        #     tracker_state,
        # )
        #
        # ====================================================

        else:

            # ------------------------------------------------
            # PLACEHOLDER PREDICTION
            # ------------------------------------------------
            #
            # Replace with your prediction.
            #
            # Format:
            #   [x, y, w, h]
            #
            # ------------------------------------------------

            pred_box = [0, 0, 0, 0]

        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        x, y, w, h = pred_box

        predictions.append({
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

        frame_idx += 1

    cap.release()

    return predictions