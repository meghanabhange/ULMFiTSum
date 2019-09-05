import logging

import fire
import gdown
import sentencepiece as spm
from fastai import *
from fastai.text import *
from gensim.corpora import WikiCorpus

from src.utils import LangTokenizer

logger = logging.getLogger()


class WikiTrainer:
    def __init__(self):
        logger.info("I work--for now")

    def load_data_lm(self):
        """
        Loads data_lm for wikipedia data.
        """
        tokenizer = Tokenizer(tok_func=LangTokenizer, lang=self.lang)
        if (self.path / "data_save.pkl").exists():
            data_lm = load_data(self.path)
        else:
            data_lm = TextLMDataBunch.from_csv(
                self.path,
                f"{self.lang}_wiki.txt",
                tokenizer=tokenizer,
                bs=48,
                text_cols=0,
                label_cols=None,
            )
            data_lm.save()
        self.data_lm = data_lm

    def train(self, lr=5e-4, n_epocs=1, fname_out="idwiki-latest"):
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        learn.fit_one_cycle(n_epocs, lr)
        learn.save(fname)

    def predict(
        self,
        start: str,
        next_tok: int,
        model_name: str,
        path="./data",
        encoder=True,
        lang="id",
    ):
        """
        Predicts the next_tok(int) after a given seed string start
        
        Arguments:
            start {str} -- initial seed string
            next_tok {int} -- number of tokens to be predicted
            model_name {str} -- Saved Language Model or Encoder
        
        Keyword Arguments:
            encoder {bool} -- Use saved encoder to predict if true (default: {True})
        
        Returns:
            [str] -- Ouput predicted string.
       """
        self.path = Path(path)
        self.lang = lang
        self.load_data_lm()
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        if encoder:
            learn.load_encoder(model_name)
        else:
            learn.load(model_name)
        output_text = learn.predict(start, next_tok)
        return output_text.replace("▁", "")


def main():
    fire.Fire(WikiTrainer)


if __name__ == "__main__":
    main()
