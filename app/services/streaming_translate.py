from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def translate_chunk(text, source_lang, target_lang):

    tokenizer.src_lang = source_lang

    inputs = tokenizer(text, return_tensors="pt")

    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)

    tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=256
    )

    return tokenizer.decode(tokens[0], skip_special_tokens=True)
