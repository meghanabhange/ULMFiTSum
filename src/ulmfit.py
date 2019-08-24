from fastai.text import *
import sentencepiece as spm
from fastai.callbacks import *
import pandas as pd
import logging

logging.getLogger().setLevel(logging.INFO)


class ULMFiT:
    def __init__(self):
        logging.info("ULMFiT model")

    def load_encoder(self, dataset_name):
        path = Path("./data")
        data_lm_path = f'data_lm_{dataset_name.lower()}.pkl'
        if not (path/data_lm_path).exists():
            data_lm = TextLMDataBunch.from_csv(path, f"{dataset_name.lower()}_train.csv")
            data_lm.save(data_lm_path)
        else:
            logging.info(f"LM data pickle exists, loading {data_lm_path}")
            data_lm = load_data(path, data_lm_path)
        logging.info("data_lm loaded")
        # Create a LM learner
        self.learn = language_model_learner(data_lm, AWD_LSTM)
        if (path/f"{dataset_name.lower()}_enc").exists():
            self.learn.load_encoder(f"{dataset_name.lower()}_enc")
        else:
            logging.info("Encoder doesn't exist, Training the encoder")
            self.train_lm(1, 1e-3)
        logging.info("encoder loaded")

    def train_lm(self, n_epocs, lr):
        self.learn.fit_one_cycle(n_epocs, lr)
        learn.save_encoder(f"{dataset_name.lower()}_enc")



def main():
    ulmfit = ULMFiT()
    ulmfit.load_encoder("Indosum")
    ulmfit.train_lm(1, 1e-3)


if __name__ == "__main__":
    main()
