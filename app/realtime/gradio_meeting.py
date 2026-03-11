import gradio as gr
import numpy as np
import soundfile as sf
import tempfile

from app.services.asr import transcribe
from app.services.translate import translate
from app.services.tts import generate_speech


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



def realtime_translate(audio, target_lang):

    if audio is None:
        return None

    sr, data = audio

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, data, sr)

    text = transcribe(temp.name)

    translated = translate(text, languages[target_lang])

    speech = generate_speech(translated)

    return speech


with gr.Blocks() as demo:

    gr.Markdown("# 🎙 Real-Time Multilingual Conversation")

    lang = gr.Dropdown(
        list(languages.keys()),
        value="Hindi",
        label="Target Language"
    )

    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy"
    )

    output_audio = gr.Audio()

    mic.stream(
        realtime_translate,
        inputs=[mic, lang],
        outputs=output_audio
    )

demo.launch(share=True)
