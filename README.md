# ULMFiTSum

Summarisation Using ULMFiT Hopefully?

## Download Data using Data Handler

```bash
    python3 -m src.download --filename all

    or 

    python3 -m src.download --filename indosum
    python3 -m src.download --filename indo_lm
    python3 -m src.download --filename sentencepiece
    python3 -m src.download --filename idwiki --lang id
```

## wiki_ulmfit

Predict

```bash
    python3 -m src.wiki_ulmfit predict --start "Alex Turner" --next_tok 10 --model_name indosum_encoder.enc
```

Finetune

```bash
    python3 -m src.wiki_ulmfit finetune --pretrained_model_name idwiki_encoder.enc --databunch_fname data_save.pkl --lr 5e-5
```