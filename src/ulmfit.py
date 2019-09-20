import logging
import fire
import gdown
import sentencepiece as spm
from fastai import *
from fastai.callbacks import *
from fastai.text import *
from gensim.corpora import WikiCorpus

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_LM_databunch(path, filename, bs=64, *flags, **params):
    if "use_sentencepiece" in flags:
        tokenizer = Tokenizer(tok_func=LangTokenizer, lang=lang)
    else:
        tokenizer = None

    if "from_folder" in flags:
        if not (path / f"{filename}.pkl").exists() or "force" in flags:
            databunch = TextLMDataBunch.from_folder(
                path, filename, tokenizer=tokenizer, bs=bs
            )
            if "save" in flags:
                databunch.save(file=params["output_filename"])
        else:
            databunch = load_data(path=path, file=filename)
    return databunch

    if "from_csv" in flags:
        if not (path / f"{filename}.pkl").exists() or "force" in flags:
            databunch = TextLMDataBunch.from_csv(
                path, filename, tokenizer=tokenizer, bs=bs
            )
            if "save" in flags:
                databunch.save(file=params["output_filename"])
        else:
            databunch = load_data(path=path, file=filename)
    return databunch


class ULMFIT:
    def __init__(self, lang, path):
        self.lang = "lang"
        self.path = path
        self.learn = None

    def language_model(self, databunch, *flags, **params):
        if not self.learn:
            learn = language_model_learner(
                databunch,
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
        if "load_lm" in flags:
            learn.load(params[model_name])
        if "load_encoder" in flags:
            learn.load_encoder(params[model_name])
        self.learn = learn

    def finetune(self, **finetune_params):
        self.learn.fit_one_cycle(**finetune_params)
        self.learn.save(f"{finetune_params['cyc_len']}_model")
