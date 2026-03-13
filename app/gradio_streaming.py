import gradio as gr
from pipelines.streaming_pipeline import run_pipeline


def stream(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_pipeline(data, sr, target)


with gr.Blocks() as demo:

    gr.Markdown("# Real-Time Speech Interpreter")

    target_lang = gr.Dropdown( 
        list(languages.keys()), 
        value="Hindi" )

    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy"
    )

    transcript = gr.Textbox(label="Transcript")
    translation = gr.Textbox(label="Translation")

    audio_out = gr.Audio(autoplay=True)

    mic.stream(
        stream,
        inputs=[mic, target],
        outputs=[transcript, translation, audio_out]
    )

demo.launch(share = True)
