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


def streaming_mode(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_pipeline(data, sr, target)


def sentence_mode(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_travel_pipeline(data, sr, target)



with gr.Blocks() as demo:

    gr.Markdown("# 🌍 Real-Time Speech Interpreter")

    with gr.Row():

        mode = gr.Radio(
            ["Streaming Interpreter", "Sentence Translator"],
            value="Streaming Interpreter",
            label="Mode"
        )

        target = gr.Dropdown(
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

    transcript = gr.Textbox(label="Transcript")

    translation = gr.Textbox(label="Translation")

    audio_out = gr.Audio(autoplay=True)



    def router(audio, target, mode):

        if mode == "Streaming Interpreter":
            return streaming_mode(audio, target)

        else:
            return sentence_mode(audio, target)



    mic.stream(
        router,
        inputs=[mic, target, mode],
        outputs=[transcript, translation, audio_out]
    )


demo.launch(share=True)
