from fastai import *
from fastai.text import *
import logging
import gdown
from gensim.corpora import WikiCorpus
from src.utils import LangTokenizer
import sentencepiece as spm

logger = logging.getLogger()


class WikiTrainer:
    # Todo Script to download sentencepiece from google drive
    def __init__(self, path):
        logger.info("heylo")
        self.path = path

    def download_indo_wiki(self):
        # Todo Figure out why this isn't working
        gdown.download(
            "https://drive.google.com/open?id=10P95rNlMPuHyB57k40KFcZUlaXdFopUC",
            f"{self.path}/0000000.txt",
            quiet=False,
        )

    def load_data_lm(self):
        tokenizer = Tokenizer(tok_func=LangTokenizer, lang="id")
        if (self.path / "data_save.pkl").exists():
            data_lm = load_data(self.path)
        else:
            data_lm = TextLMDataBunch.from_csv(
                self.path,
                "0000000.txt",
                tokenizer=tokenizer,
                bs=48,
                text_cols=0,
                label_cols=None,
            )
            data_lm.save()
        self.data_lm = data_lm

    def train(self, lr=5e-4, n_epocs=1, fname_out="idwiki-latest"):
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        learn.fit_one_cycle(1, lr)
        learn.unfreeze()
        learn.fit_one_cycle(n_epocs, lr)
        learn.save(fname)

    def predict(self, sent, n_words, latest_checkpoint_name="idwiki-latest"):
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        learn.load(latest_checkpoint_name)
        output_text = learn.predict(sent, n_words)
        return output_text

    def predict_encoder(self, sent, n_words, encoder_name):
        learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.3)
        learn.load_encoder(encoder_name)
        output_text = learn.predict(sent, n_words)
        return output_text


wiki_trainer = WikiTrainer(Path("./data"))
# wiki_trainer.download_indo_wiki()
wiki_trainer.load_data_lm()
# wiki_trainer.train(fname_out='idwiki-testing')
output_text = wiki_trainer.predict("▁melemahkan", 20, "idwiki-8")
print(output_text)
output_text = wiki_trainer.predict_encoder(
    "_melemahkan", 20, "indowiki_encoder_attempt1.enc"
)
print(output_text)
