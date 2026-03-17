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


def interpreter(audio, target):

    if audio is None:
        return "", "", None, ""

    sr, data = audio

    text, translated, speech, timings = run_pipeline(data, sr, target)

    latency = f"""
ASR: {timings.get('asr',0)} ms
Translation: {timings.get('translation',0)} ms
TTS: {timings.get('tts',0)} ms
Total: {timings.get('total',0)} ms
"""

    return text, translated, speech, latency


def translator(audio, target):

    if audio is None:
        return "", "", None, ""

    sr, data = audio

    text, translated, speech, timings = run_travel_pipeline(data, sr, target)

    latency = f"""
ASR: {timings.get('asr',0)} ms
Translation: {timings.get('translation',0)} ms
TTS: {timings.get('tts',0)} ms
Total: {timings.get('total',0)} ms
"""

    return text, translated, speech, latency


with gr.Blocks() as demo:

    gr.Markdown("# 🌍 AI Speech Interpreter + Translator")

    # -------- Interpreter --------
    with gr.Tab("Speech Interpreter (Live)"):

        target_i = gr.Dropdown(list(languages.keys()), value="Hindi")

        mic_stream = gr.Audio(sources=["microphone"], streaming=True, type="numpy")

        transcript_i = gr.Textbox(label="Transcript")
        translation_i = gr.Textbox(label="Translation")
        audio_i = gr.Audio(autoplay=True)
        latency_i = gr.Textbox(label="Latency")

        mic_stream.stream(
            interpreter,
            inputs=[mic_stream, target_i],
            outputs=[transcript_i, translation_i, audio_i, latency_i]
        )

    # -------- Translator --------
    with gr.Tab("Speech Translator (Record)"):

        target_t = gr.Dropdown(list(languages.keys()), value="Hindi")

        mic_record = gr.Audio(sources=["microphone"], type="numpy")

        btn = gr.Button("Translate & Dub")

        transcript_t = gr.Textbox(label="Transcript")
        translation_t = gr.Textbox(label="Translation")
        audio_t = gr.Audio(autoplay=True)
        latency_t = gr.Textbox(label="Latency")

        btn.click(
            translator,
            inputs=[mic_record, target_t],
            outputs=[transcript_t, translation_t, audio_t, latency_t]
        )


demo.launch(share=True)
