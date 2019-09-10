import logging

import fire
import gdown
import sentencepiece as spm
from fastai import *
from fastai.callbacks import *
from fastai.text import *
from gensim.corpora import WikiCorpus

from src.utils import LangTokenizer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
    def __init__(
        self,
        lr=5e-4,
        n_epocs=1,
        fname_out="idwiki-latest",
        load_pretrained=False,
        path="./data",
        lang="id",
    ):
        self.lr = lr
        self.n_epocs = n_epocs
        self.fname_out = fname_out
        self.load_pretrained = load_pretrained
        self.path = Path(path)
        self.lang = lang

    def finetune(
        self,
        pretrained_model_name,
        databunch_fname,
        path="./data",
        dataset="indosum",
        n_epocs=1,
        fname_out="finetune_lm_latest",
        **finetune_args,
    ):
        path = Path(path)
        if not (path / "models" / f"{pretrained_model_name}.pth").exists():
            logger.info(
                f"{pretrained_model_name} does not exist, check {path/'models'/f'{pretrained_model_name}.pth'}"
            )
            return
        if not (path / databunch_fname).exists():
            logger.info(
                f"{databunch_fname} does not exist, check {path/databunch_fname}"
            )
            return
        learn = self.load_language_model(
            path=path,
            model_name=pretrained_model_name,
            encoder=True,
            load_pretrained=True,
            databunch_fname=databunch_fname,
        )
        learn.fit_one_cycle(n_epocs, **finetune_args)
        learn.save(fname_out)

    def fit(self):
        learn = self.load_language_model(
            path=self.path, load_pretrained=self.load_pretrained
        )
        learn.fit_one_cycle(self.n_epocs, self.lr)
        learn.save(self.fname_out)

    def load_language_model(
        self,
        path,
        model_name="idwiki_encoder.enc",
        encoder=True,
        load_pretrained=True,
        databunch_fname="data_save.pkl",
    ):
        path = Path(path)
        self.data_lm = load_data_file(
            path=path,
            fname_pkl=databunch_fname,
            fname_text_file=f"{self.lang}_wiki.txt",
            lang=self.lang,
        )
        learn = language_model_learner(
            self.data_lm,
            AWD_LSTM,
            drop_mult=0.3,
            metrics=[accuracy, Perplexity()],
            callback_fns=[
                partial(
                    EarlyStoppingCallback,
                    monitor="Perplexity()",
                    min_delta=2,
                    patience=3,
                )
            ],
        )
        if not load_pretrained:
            return learn
        if encoder:
            learn.load_encoder(model_name)
        else:
            learn.load(model_name)
        return learn

    def predict(
        self,
        start: str,
        next_tok: int,
        model_name="idwiki_encoder.enc",
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
        path = Path(path)
        self.lang = lang
        learn = self.load_language_model(
            path=path, model_name=model_name, encoder=encoder
        )
        output_text = learn.predict(start, next_tok, **kwargs)
        return output_text.replace("▁", "")


def main():
    fire.Fire(WikiTrainer)


if __name__ == "__main__":
    main()
