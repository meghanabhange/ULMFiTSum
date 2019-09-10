from pathlib import Path

import fire
import gdown

from src.utils import DataHandler


def get_filename(
    data_path=Path("./data"), filename="indosum", wiki_text=False, lang="id"
):
    return data_path, filename, wiki_text, lang


def download_wiki_files(path, wiki_text=False, lang="id"):
    """
        Downloads Processed Wikipedia from google drive. 
    """
    lang_dict = {
        "id": {
            "wiki": "https://drive.google.com/uc?export=download&id=10P95rNlMPuHyB57k40KFcZUlaXdFopUC",
            "data": "https://drive.google.com/uc?export=download&id=1-0gaC0bmYyMoUWIkGoH6fY6DUQTArMD3",
        }
    }
    if not (path / "data_save.pkl").exists():
        gdown.download(lang_dict[lang]["data"], f"{path}/data_save.pkl", quiet=False)
    if wiki_text:
        gdown.download(lang_dict[lang["wiki"]], f"{path}/{lang}_wiki.txt", quiet=False)


def main():
    data_path, filename, wiki_text, lang = fire.Fire(get_filename)
    if filename == "idwiki":
        download_wiki_files(data_path, wiki_text, lang)
    elif filename == "all":
        data_handler = DataHandler(data_path)
        data_handler.downloader("indosum")
        data_handler.downloader("indo_lm")
        data_handler.downloader("sentencepiece")
        download_wiki_files(data_path, wiki_text, lang)
    else:
        data_handler = DataHandler(data_path)
        data_handler.downloader(filename)


if __name__ == "__main__":
    main()
