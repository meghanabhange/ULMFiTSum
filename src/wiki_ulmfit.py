import logging

import fire
import gdown
import sentencepiece as spm
from fastai import *
from fastai.text import *
from gensim.corpora import WikiCorpus

from src.utils import LangTokenizer

logger = logging.getLogger()


def load_data_file(
    path, fname_pkl="data_save.pkl", fname_text_file="id_wiki.txt", lang="id"
):
    tokenizer = Tokenizer(tok_func=LangTokenizer, lang=lang)
    if (path / fname_pkl).exists():
        data_file = load_data(path)
    else:
        data_file = TextLMDataBunch.from_csv(
            path,
            fname_text_file,
            tokenizer=tokenizer,
            bs=48,
            text_cols=0,
            label_cols=None,
        )
        data_file.save()

    return data_file


class WikiTrainer:
    def __init__(self):
        logger.info("I work--for now")

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
        **kwargs,
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
        self.data_lm = load_data_file(
            path=self.path,
            fname_pkl="data_save.pkl",
            fname_text_file=f"{self.lang}_wiki.txt",
            lang=self.lang,
        )
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        if encoder:
            learn.load_encoder(model_name)
        else:
            learn.load(model_name)
        output_text = learn.predict(start, next_tok, **kwargs)
        return output_text.replace("▁", "")


def main():
    fire.Fire(WikiTrainer)


if __name__ == "__main__":
    main()
