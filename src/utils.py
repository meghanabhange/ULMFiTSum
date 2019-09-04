"""
Let's start by creating some basic utilites
that are required for Summarisation.

1. Downloading the data for Indosum
"""
import logging
import subprocess
from pathlib import Path
from fastai import *
from fastai.text import *
import sentencepiece as spm

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


class LangTokenizer(BaseTokenizer):
    def __init__(
        self, lang: str, vocab_size: int = 60000, path_to_sp=f"data/sentencepiece/"
    ):
        self.lang = lang
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path_to_sp + f"{lang}_lm.model")
        self.vocab = Vocab(
            [self.sp.IdToPiece(int(i)) for i in range(self.vocab_size)]
        )  # Read about what this does exactly

    def tokenizer(self, t: str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

    def detokenizer(self, t: List[str]) -> str:
        return self.sp.DecodePieces(t)
