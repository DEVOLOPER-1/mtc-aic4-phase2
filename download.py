"""
Download Module

A brief description of what this module does.

This module provides:
- Downloading files from Google Drive
- Managing checkpoint directories
- Checking for existing files before download

You are required to replace "YOUR_FILE_ID" with the actual file ID from Google Drive to download the checkpoint.
Or replace the download method with your preferred method if you are not using Google Drive.

"""

import os
import gdown


def download_checkpoint(file_id, checkpoint_dir="./checkpoints"):
    """
    Download checkpoint if it does not exist.

    Recommended:
        Google Drive + gdown

    Return:
        checkpoint_path
    """

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "model.pth",
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    if os.path.exists(checkpoint_path):

        print("Checkpoint already exists.")

        return checkpoint_path


    # Example link:
    # https://drive.google.com/file/d/FILE_ID/view
    # put your file id instead

    url = (
        f"https://drive.google.com/uc?id={file_id}"
    )

    gdown.download(
        url,
        checkpoint_path,
        quiet=False,
    )

    return checkpoint_path

if __name__ == "__main__":
    download_checkpoint("YOUR_FILE_ID")
