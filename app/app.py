import gradio as gr

from services.asr import transcribe
from services.translate import translate
from services.tts import generate_speech

# Supported target languages
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


def process(audio, lang):

    if audio is None:
        return "Please record audio", "", None

    try:

        # Speech → Text
        text = transcribe(audio)

        # Translate text
        translated = translate(text, languages[lang])

        # Call TTS microservice
        audio_file = generate_speech(translated)

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
