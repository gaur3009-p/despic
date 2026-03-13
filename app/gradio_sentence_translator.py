import gradio as gr
from pipelines.sentence_pipeline import run_travel_pipeline

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

def translate_sentence(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_travel_pipeline(data, sr, target)


with gr.Blocks() as demo:

    gr.Markdown("# Travel Speech Translator")

    target = gr.Dropdown(
        language = list(language.keys()),
        value="Hindi"
    )

    mic = gr.Audio(
        sources=["microphone"],
        type="numpy"
    )

    translate_btn = gr.Button("Translate")

    transcript = gr.Textbox(label="Transcript")
    translation = gr.Textbox(label="Translation")

    audio_out = gr.Audio(autoplay=True)

    translate_btn.click(
        translate_sentence,
        inputs=[mic, target],
        outputs=[transcript, translation, audio_out]
    )

demo.launch()
