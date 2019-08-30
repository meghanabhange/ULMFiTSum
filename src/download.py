from src.utils import DataHandler
from pathlib import Path
import fire


def get_dataset(data_path=Path("./data"), dataset="indosum", pretrained="indo_lm"):
    return data_path, dataset, pretrained


def main():
    data_path, dataset, pretrained= fire.Fire(get_dataset)
    data_handler = DataHandler(data_path)
    # data_handler.downloader(dataset)
    if pretrained=="indo_lm":
        data_handler.downloader("indo_lm")


if __name__ == "__main__":
    main()
