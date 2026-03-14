import gradio as gr

from pipelines.streaming_pipeline import run_pipeline
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


# --------------------------
# LIVE INTERPRETER
# --------------------------

def interpreter(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_pipeline(data, sr, target)



# --------------------------
# RECORD TRANSLATOR
# --------------------------

def translator(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_travel_pipeline(data, sr, target)



with gr.Blocks() as demo:

    gr.Markdown("# 🌍 AI Speech Interpreter + Translator")

    with gr.Row():

        mode = gr.Radio(
            ["Speech Interpreter (Live)", "Speech Translator (Record)"],
            value="Speech Interpreter (Live)",
            label="Mode"
        )

        target = gr.Dropdown(
            list(languages.keys()),
            value="Hindi",
            label="Target Language"
        )


    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label="Speak"
    )


    transcript = gr.Textbox(label="Transcript")

    translation = gr.Textbox(label="Translation")

    audio_out = gr.Audio(
        autoplay=True,
        label="Translated Speech"
    )


    # --------------------------
    # INTERPRETER STREAM
    # --------------------------

    mic.stream(
        interpreter,
        inputs=[mic, target],
        outputs=[transcript, translation, audio_out]
    )


    # --------------------------
    # TRANSLATOR RECORD MODE
    # --------------------------

    mic.change(
        translator,
        inputs=[mic, target],
        outputs=[transcript, translation, audio_out]
    )


demo.launch(share=True)
