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


# -----------------------------
# SPEECH INTERPRETER (LIVE)
# -----------------------------

def interpreter(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_pipeline(data, sr, target)



# -----------------------------
# SPEECH TRANSLATOR (RECORD)
# -----------------------------

def translator(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_travel_pipeline(data, sr, target)



# -----------------------------
# UI
# -----------------------------

with gr.Blocks() as demo:

    gr.Markdown("# 🌍 AI Speech Interpreter + Translator")

    # =================================
    # TAB 1 : LIVE INTERPRETER
    # =================================

    with gr.Tab("Speech Interpreter (Live)"):

        gr.Markdown("### Real-time conversation translation")

        target_interpreter = gr.Dropdown(
            list(languages.keys()),
            value="Hindi",
            label="Target Language"
        )

        mic_stream = gr.Audio(
            sources=["microphone"],
            streaming=True,
            type="numpy",
            label="Speak"
        )

        transcript_stream = gr.Textbox(label="Transcript")

        translation_stream = gr.Textbox(label="Translation")

        audio_stream = gr.Audio(
            autoplay=True,
            label="Translated Speech"
        )

        mic_stream.stream(
            interpreter,
            inputs=[mic_stream, target_interpreter],
            outputs=[transcript_stream, translation_stream, audio_stream]
        )


    # =================================
    # TAB 2 : SPEECH TRANSLATOR
    # =================================

    with gr.Tab("Speech Translator (Record)"):

        gr.Markdown("### Record speech and translate")

        target_translator = gr.Dropdown(
            list(languages.keys()),
            value="Hindi",
            label="Target Language"
        )

        mic_record = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Record Speech"
        )

        transcript_record = gr.Textbox(label="Transcript")

        translation_record = gr.Textbox(label="Translation")

        audio_record = gr.Audio(
            autoplay=True,
            label="Translated Speech"
        )

        mic_record.change(
            translator,
            inputs=[mic_record, target_translator],
            outputs=[transcript_record, translation_record, audio_record]
        )


demo.launch(share=True)
