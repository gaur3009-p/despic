import gradio as gr
import tempfile
import soundfile as sf

from services.asr import transcribe
from services.translate import translate
from services.tts import generate_speech
from realtime.vad import get_speech_segments


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


transcript_log = []
translation_log = []


def realtime_pipeline(audio, target_lang):

    if audio is None:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    sr, data = audio

    # Detect speech
    speech_segments = get_speech_segments(data, sr)

    if len(speech_segments) == 0:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    segment = speech_segments[0]

    speech_audio = data[segment["start"]:segment["end"]]

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, speech_audio, sr)

    # ASR
    text = transcribe(temp.name)

    if text.strip() == "":
        return "\n".join(transcript_log), "\n".join(translation_log), None

    transcript_log.append(text)

    # Translation
    translated = translate(text, languages[target_lang])

    translation_log.append(translated)

    # Auto TTS (immediate dubbing)
    speech = generate_speech(translated)

    return "\n".join(transcript_log), "\n".join(translation_log), speech


def clear_conversation():

    transcript_log.clear()
    translation_log.clear()

    return "", "", None


with gr.Blocks() as demo:

    gr.Markdown("# 🎙 Real-Time Multilingual Speech Translator")

    target_lang = gr.Dropdown(
        list(languages.keys()),
        value="Hindi",
        label="Target Language"
    )

    mic = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Speak (Auto Translate + Dub)"
    )

    with gr.Row():

        original_text = gr.Textbox(
            label="Live Transcript",
            lines=12
        )

        translated_text = gr.Textbox(
            label="Live Translation",
            lines=12
        )

    translated_audio = gr.Audio(label="Dubbed Translation")

    clear_btn = gr.Button("Clear Conversation")

    mic.stream(
        realtime_pipeline,
        inputs=[mic, target_lang],
        outputs=[original_text, translated_text, translated_audio]
    )

    clear_btn.click(
        clear_conversation,
        outputs=[original_text, translated_text, translated_audio]
    )

demo.launch(share=True)
