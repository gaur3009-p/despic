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

# Conversation memory
transcript_log = []
translation_log = []


def realtime_pipeline(audio, target_lang):

    if audio is None:
        return "", "", None

    sr, data = audio

    # Detect speech segments using VAD
    speech_segments = get_speech_segments(data, sr)

    if len(speech_segments) == 0:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    # Take the first detected speech segment
    segment = speech_segments[0]

    speech_audio = data[segment["start"]:segment["end"]]

    # Save temporary audio
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

    # TTS for current sentence
    speech = generate_speech(translated)

    return "\n".join(transcript_log), "\n".join(translation_log), speech


def play_full_translation():

    if len(translation_log) == 0:
        return None

    full_text = " ".join(translation_log)

    speech = generate_speech(full_text)

    return speech


def clear_conversation():

    transcript_log.clear()
    translation_log.clear()

    return "", "", None


with gr.Blocks() as demo:

    gr.Markdown("# 🎙 Real-Time Multilingual Conversation (Phase-3)")

    with gr.Row():

        target_lang = gr.Dropdown(
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

    with gr.Row():

        original_text = gr.Textbox(
            label="Live Transcript (ASR)",
            lines=12
        )

        translated_text = gr.Textbox(
            label="Live Translation",
            lines=12
        )

    translated_audio = gr.Audio(label="Translated Speech (Latest)")

    with gr.Row():

        play_all = gr.Button("🔊 Play Full Translated Conversation")

        clear_btn = gr.Button("🗑 Clear Conversation")

    full_audio = gr.Audio(label="Full Conversation Translation")

    mic.stream(
        realtime_pipeline,
        inputs=[mic, target_lang],
        outputs=[original_text, translated_text, translated_audio]
    )

    play_all.click(
        play_full_translation,
        outputs=full_audio
    )

    clear_btn.click(
        clear_conversation,
        outputs=[original_text, translated_text, translated_audio]
    )


demo.launch(share=True)
