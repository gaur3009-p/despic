from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

# IMPORTANT: trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True
).to(device)


def translate(text, src_lang, tgt_lang):

    inputs = tokenizer(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=1
    )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated
