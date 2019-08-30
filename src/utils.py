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
        if not self.check_data_exists(download_type):
            (self.data_path / download_type).mkdir(parents=True, exist_ok=True)
        else:
            logging.info("Folder already exists")
        if not (self.data_path / download_type / f"{download_type}.tar.gz").exists():
            google_drive_link = {
                "indosum": "https://docs.google.com/uc?export=download&id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco",
                "indo_lm": "link_to_pretrained_gooogle_lm",
            }
            output_download = subprocess.check_output(
                [
                    "wget",
                    "--no-check-certificate",
                    google_drive_link[download_type],
                    "-O",
                    f"{self.data_path/download_type/f'{download_type}.tar.gz'}",
                ]
            )
            logging.info(f"{output_download}")
        if not (self.data_path / download_type / f"{download_type}").exists():
            ouput_unzip = subprocess.check_output(
                [
                    "tar",
                    "xvzf",
                    f"{self.data_path/download_type/f'{download_type}.tar.gz'}",
                    "-C",
                    f"{self.data_path/download_type}",
                ]
            )
            logging.info(f"{ouput_unzip}")
        logging.info(f"{download_type} Downloaded")

    def check_data_exists(self, folder_name):
        """
        Checks the data folders existence

        Arguments:
            folder_name {[str]} -- [Name of the folder to be downloaded]

        Returns:
            [bool] -- [True is the data folder exists]
        """
        logging.info(f"Checking for : {folder_name}")
        exists = (self.data_path / folder_name).exists()
        logging.info(f"{folder_name} Folder Existence status : {exists}")
        return exists
