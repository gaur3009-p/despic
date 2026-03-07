import gradio as gr

from services.asr import transcribe
from services.translate import translate


languages = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn"
}


def process(audio, lang):

    if audio is None:
        return "Please record audio", ""

    text = transcribe(audio)

    translated = translate(text, languages[lang])

    return text, translated


ui = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", sources=["microphone"]),
        gr.Dropdown(list(languages.keys()), label="Target Language")
    ],
    outputs=[
        gr.Textbox(label="Original Text"),
        gr.Textbox(label="Translated Text")
    ],
    title="Phase 1 – Speech Translator"
)

ui.launch(share = True)
