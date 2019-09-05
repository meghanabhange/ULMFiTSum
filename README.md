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

```python
    wiki_trainer = WikiTrainer(Path("./data"))
    wiki_trainer.load_data_lm()
    output_text = wiki_trainer.predict(
        next_tok=20,
        start="Saya Meghana, Insinyur Pembelajaran dan Bahasa Dalam",
        model_name="idwiki_encoder.enc",
        enocder = True
    )
    print(output_text)
```