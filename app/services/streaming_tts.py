import torch
import soundfile as sf
import uuid

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts"
).to(device)

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)


def generate_streaming_speech(text):

    description = (
        "Divya speaks clearly with a natural conversational tone."
    )

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt"
    ).to(device)

    prompt_inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask
    )

    audio = generation.cpu().numpy().squeeze()

    file = f"stream_{uuid.uuid4()}.wav"

    sf.write(file, audio, model.config.sampling_rate)

    return file
