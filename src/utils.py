"""
Let's start by creating some basic utilites
that are required for Summarisation.

1. Downloading the data for Indosum
"""
import logging
import subprocess
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)


class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        logging.info(f"Data Handler Class Created - {data_path}")

    def downloader(self, download_type):
        """
        Downaloads download_type in data_path/download_type.
        """
        logging.info(f"Downloading {download_type}")
        if not (self.data_path / f"{download_type}.tar.gz").exists():
            google_drive_link = {
                "indosum": "https://docs.google.com/uc?export=download&id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco",
                "indo_lm": "https://drive.google.com/uc?export=download&id=1tWWi4TFTlNLFKtHGnQqAURKE0r8DPK9E",
            }
            output_download = subprocess.check_output(
                [
                    "wget",
                    "--no-check-certificate",
                    google_drive_link[download_type],
                    "-O",
                    f"{self.data_path/f'{download_type}.tar.gz'}",
                ]
            )
            logging.info(f"{output_download}")
        if not (self.data_path / f"{download_type}").exists():
            ouput_unzip = subprocess.check_output(
                [
                    "tar",
                    "xvzf",
                    f"{self.data_path/f'{download_type}.tar.gz'}",
                    "-C",
                    f"{self.data_path}",
                ]
            )
            logging.info(f"{ouput_unzip}")
        logging.info(f"{download_type} Downloaded")