from src.utils import DataHandler
from pathlib import Path
import fire


def get_dataset(dataset="indosum"):
    return dataset


def main():
    data_path = Path("./data")
    dataset = fire.Fire(get_dataset)
    data_handler = DataHandler(data_path, "indosum")


if __name__ == "__main__":
    main()
