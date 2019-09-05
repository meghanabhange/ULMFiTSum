from pathlib import Path

import fire

from src.utils import DataHandler


def get_filename(data_path=Path("./data"), filename="indosum"):
    return data_path, filename


def main():
    data_path, filename = fire.Fire(get_filename)
    data_handler = DataHandler(data_path)
    data_handler.downloader(filename)


if __name__ == "__main__":
    main()
