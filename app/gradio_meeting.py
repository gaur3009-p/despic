import gradio as gr
import numpy as np
import soundfile as sf
import tempfile

from services.asr import transcribe
from services.translate import translate
from services.tts import generate_speech


languages = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Sindhi": "snd_Arab",
    "Sanskrit": "san_Deva"
}


def realtime_pipeline(audio, target_lang):

    if audio is None:
        return "", "", None

    sr, data = audio

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, data, sr)

    # ASR
    text = transcribe(temp.name)

    # Translation
    translated = translate(text, languages[target_lang])

    # TTS
    speech = generate_speech(translated)

    return text, translated, speech


with gr.Blocks() as demo:

    gr.Markdown("# 🎙 Real-Time Multilingual Conversation (Phase 3)")

    with gr.Row():

        target_lang = gr.Dropdown(
            list(languages.keys()),
            value="Hindi",
            label="Target Language"
        )

    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Speak"
    )

    with gr.Row():

        original_text = gr.Textbox(
            label="Original Speech (ASR)",
            lines=4
        )

        translated_text = gr.Textbox(
            label="Translated Text",
            lines=4
        )

    translated_audio = gr.Audio(label="Translated Speech")

    mic.stream(
        realtime_pipeline,
        inputs=[mic, target_lang],
        outputs=[original_text, translated_text, translated_audio]
    )


demo.launch(share=True)
