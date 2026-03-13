import gradio as gr
from pipelines.streaming_pipeline import run_pipeline


def translate_sentence(audio, target):

    if audio is None:
        return "", "", None

    sr, data = audio

    return run_pipeline(data, sr, target)


with gr.Blocks() as demo:

    gr.Markdown("# Travel Speech Translator")

    target_lang = gr.Dropdown( 
      list(languages.keys()), 
      value="Hindi" )

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

demo.launch(share = True)
