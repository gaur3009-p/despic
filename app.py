import gradio as gr

from services.asr import transcribe
from services.translate import translate
from services.tts_client import speak


# Supported target languages
languages = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn"
}


def process(audio, lang):

    if audio is None:
        return "Please record audio", "", None

    try:

        # Speech → Text
        text = transcribe(audio)

        # Translate text
        translated = translate(text, languages[lang])

        # Call TTS microservice
        audio_file = speak(translated)

        return text, translated, audio_file

    except Exception as e:

        return f"Error: {str(e)}", "", None


ui = gr.Interface(
    fn=process,

    inputs=[
        gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak"
        ),
        gr.Dropdown(
            list(languages.keys()),
            label="Target Language"
        )
    ],

    outputs=[
        gr.Textbox(label="Original Text"),
        gr.Textbox(label="Translated Text"),
        gr.Audio(label="Translated Speech")
    ],

    title="Phase 2 – Multilingual Speech Translator"
)


if __name__ == "__main__":
    ui.launch(share = True)
