import logging

import gdown
import sentencepiece as spm
from fastai import *
from fastai.text import *
from gensim.corpora import WikiCorpus

from src.utils import LangTokenizer

logger = logging.getLogger()


class WikiTrainer:
    def __init__(self, path, lang="id"):
        self.path = path
        self.lang = lang

    def download_wiki_files(self, wiki_text=False):
        """
        Downloads Processed Wikipedia from google drive. 
        """
        lang_dict = {
            "id": {
                "wiki": "https://drive.google.com/uc?export=download&id=10P95rNlMPuHyB57k40KFcZUlaXdFopUC",
                "data": "https://drive.google.com/uc?export=download&id=1-0gaC0bmYyMoUWIkGoH6fY6DUQTArMD3",
            }
        }
        if not (self.path/'data_save.pkl').exists():
            gdown.download(
                lang_dict[self.lang]["data"], f"{self.path}/data_save.pkl", quiet=False
            )
        if wiki_text:
            gdown.download(
                lang_dict[self.lang["wiki"]],
                f"{self.path}/{self.lang}_wiki.txt",
                quiet=False,
            )

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

    def predict(self, start: str, next_tok: int, model_name: str, encoder=True):
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
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        if encoder:
            learn.load_encoder(model_name)
        else:
            learn.load(model_name)
        output_text = learn.predict(start, next_tok)
        return output_text


def main():
    wiki_trainer = WikiTrainer(Path("./data"))
    wiki_trainer.download_wiki_files()
    wiki_trainer.load_data_lm()
    output_text = wiki_trainer.predict(
        next_tok=20,
        start="Saya Meghana,",
        model_name="idwiki_encoder.enc",
    )
    print(output_text.replace("▁", " "))


if __name__ == "__main__":
    main()
