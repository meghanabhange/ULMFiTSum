import logging

import pandas as pd
import sentencepiece as spm
import torch
from fastai.callbacks import *
from fastai.text import *
from torch.autograd import Variable

download_logger = logging.getLogger(name="download_logger")
download_logger.setLevel(logging.INFO)


class ULMFiT:
    def __init__(self):
        download_logger.info("ULMFiT model")

    def load_model(self, dataset_name):
        path = Path("./data")
        data_lm_path = f"data_lm_{dataset_name.lower()}.pkl"
        data_lm = self.load_data_lm(path, data_lm_path)
        download_logger.info("data_lm loaded")
        model = get_language_model(AWD_LSTM, len(data_lm.vocab.itos))
        download_logger.info("Language Model")
        # model.reset()
        # model.eval()

    def load_lm(self, dataset_name):
        path = Path("./data")
        data_lm_path = f"data_lm_{dataset_name.lower()}.pkl"
        data_lm = self.load_data_lm(path, data_lm_path)
        download_logger.info("data_lm loaded")
        # Create a LM learner
        # learn = language_model_learner(data_lm, AWD_LSTM)
        model = get_language_model(AWD_LSTM, len(data_lm.vocab.itos))
        download_logger.info("Language Model")
        learn = self.load_encoder(learn, path, dataset_name)
        download_logger.info("encoder loaded")
        return learn

    def load_encoder(self, learn, path, dataset_name):
        if (path / f"{dataset_name.lower()}_enc").exists():
            learn.load_encoder(f"{dataset_name.lower()}_enc")
        else:
            download_logger.info("Encoder doesn't exist, Training the encoder")
            self.train_lm(learn, 1, 1e-3)
        return learn

    def load_data_lm(self, path, data_lm_path):
        if not (path / data_lm_path).exists():
            data_lm = TextLMDataBunch.from_csv(
                path, f"{dataset_name.lower()}_train.csv"
            )
            data_lm.save(data_lm_path)
        else:
            download_logger.info(f"LM data pickle exists, loading {data_lm_path}")
            data_lm = load_data(path, data_lm_path)
        return data_lm

    def train_lm(self, learn, n_epocs, lr):
        learn.fit_one_cycle(n_epocs, lr)
        learn.save_encoder(f"{dataset_name.lower()}_enc")


def main():
    ulmfit = ULMFiT()
    ulmfit.load_model("Indosum")


if __name__ == "__main__":
    main()
