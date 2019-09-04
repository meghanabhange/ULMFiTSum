"""
Let's start by creating some basic utilites
that are required for Summarisation.

1. Downloading the data for Indosum
"""
import logging
import subprocess
from pathlib import Path

import gdown

download_logger = logging.getLogger(name="download_logger")
download_logger.setLevel(logging.INFO)


class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        download_logger.info(f"Data Handler Class Created - {data_path}")
        self.data_path.mkdir(parents=True, exist_ok=True)

    def downloader(self, download_type):
        """
        Downloads download_type in data_path/download_type.
        """
        download_logger.info(f"Downloading {download_type}")
        fname = f"{download_type}.tar.gz"
        if not (self.data_path / fname).exists():
            google_drive_link = {
                "indosum": "https://drive.google.com/uc?export=download&id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco",
                "indo_lm": "https://drive.google.com/uc?export=download&id=14uhX9s43eKAsy7b94FV5mHn0vDeJc4YH",
            }
            url = google_drive_link[download_type]
            output = f"{self.data_path/fname}"
            gdown.download(url, output, quiet=False)
        if not (self.data_path / f"{download_type}").exists():
            output_unzip = subprocess.check_output(
                ["tar", "xvzf", f"{self.data_path/fname}", "-C", f"{self.data_path}"]
            )
            download_logger.info(f"{output_unzip}")
        download_logger.info(f"{download_type} Downloaded")
