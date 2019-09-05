# ULMFiTSum

Summarisation Using ULMFiT Hopefully?

## Download Data using Data Handler

```bash
    python3 -m src.download --filename indosum
    python3 -m src.download --filename indo_lm
    python3 -m src.download --filename sentencepiece
    python3 -m src.download --filename idwiki --lang id
```

## wiki_ulmfit

```bash
    python3 -m src.wiki_ulmfit predict --start "Saya Megs" --next_tok 10 --model_name idwiki_encoder.enc
```