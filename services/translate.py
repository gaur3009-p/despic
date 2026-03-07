from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def translate(text, target_lang):

    # tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # get language token id correctly
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)

    # generate translation
    tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id
    )

    # decode result
    return tokenizer.decode(tokens[0], skip_special_tokens=True)
