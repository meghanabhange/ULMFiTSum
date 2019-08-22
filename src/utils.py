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
    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name
        logging.info(f"Data Handler Class Created - {data_path}")
        if dataset_name == "Indosum":
            self.download_indosum()

    def download_indosum(self):
        """
        Downaloads Indosum in data_path/indosum.
        """
        logging.info("Downloading Indosum")
        if not self.check_data_exists(self.dataset_name):
            (self.data_path / self.dataset_name).mkdir(parents=True, exist_ok=True)
        else:
            logging.info("Folder already exists")
        if not (self.data_path / self.dataset_name / "indosum.tar.gz").exists():
            output_download = subprocess.check_output(
                [
                    "wget",
                    "--no-check-certificate",
                    "https://docs.google.com/uc?export=download&id=1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco",
                    "-O",
                    f"{self.data_path/self.dataset_name/'indosum.tar.gz'}",
                ]
            )
            logging.info(f"{output_download}")
        if not (self.data_path / self.dataset_name / "indosum").exists():
            ouput_unzip = subprocess.check_output(
                [
                    "tar",
                    "xvzf",
                    f"{self.data_path/self.dataset_name/'indosum.tar.gz'}",
                    "-C",
                    f"{self.data_path/self.dataset_name}",
                ]
            )
            logging.info(f"{ouput_unzip}")
        logging.info(f"{self.dataset_name} Downloaded")

    def check_data_exists(self, name_of_dataset):
        """
        Checks the data folders existence

        Arguments:
            name_of_dataset {[str]} -- [Name of the dataset to be downloaded]

        Returns:
            [bool] -- [True is the data folder exists]
        """
        logging.info(f"Checking for : {name_of_dataset}")
        exists = (str(self.data_path)/name_of_dataset).exists()
        logging.info(f"{name_of_dataset} Folder Existence status : {exists}")
        return exists


def main():
    data_path = Path("./data")
    data_handler = DataHandler(data_path, "Indosum")


if __name__ == "__main__":
    main()
