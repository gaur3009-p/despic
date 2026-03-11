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
    "French": "fra_Latn",
    "Spanish": "spa_Latn"
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
