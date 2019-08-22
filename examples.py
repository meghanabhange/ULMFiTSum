from src.utils import DataHandler
from pathlib import Path


def main():

    # Downloading Data

    data_path = Path("./data")
    data_handler = DataHandler(data_path, "Indosum")


if __name__ == "__main__":
    main()
