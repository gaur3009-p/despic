import gradio as gr
import tempfile
import soundfile as sf

from realtime.vad import get_speech_segments
from realtime.audio_buffer import AudioBuffer
from realtime.streaming_asr import StreamingASR

from services.streaming_translate import translate_chunk
from services.streaming_tts import generate_streaming_speech


# Initialize ASR engine
asr_engine = StreamingASR()

# Audio buffer
audio_buffer = None

# Conversation logs
transcript_log = []
translation_log = []


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


def realtime_pipeline(audio, target_lang):

    global audio_buffer
    global transcript_log
    global translation_log

    if audio is None:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    sr, data = audio

    # Initialize buffer
    if audio_buffer is None:
        audio_buffer = AudioBuffer(sr)

    # Append new audio chunk
    audio_buffer.append(data)

    # Wait until enough audio collected
    if not audio_buffer.ready():
        return "\n".join(transcript_log), "\n".join(translation_log), None

    buffer_audio = audio_buffer.get_buffer()

    # Detect speech using VAD
    speech_segments = get_speech_segments(buffer_audio, sr)

    if len(speech_segments) == 0:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    segment = speech_segments[0]

    speech_audio = buffer_audio[segment["start"]:segment["end"]]

    # Remove processed audio
    audio_buffer.trim(segment["end"])

    # Save speech chunk
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, speech_audio, sr)

    # Streaming ASR
    text, source_lang = asr_engine.transcribe(temp.name)

    if text is None or text.strip() == "":
        return "\n".join(transcript_log), "\n".join(translation_log), None

    # Prevent duplicates
    if len(transcript_log) > 0 and text == transcript_log[-1]:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    transcript_log.append(text)

    # Translation
    translated = translate_chunk(
        text,
        source_lang,
        languages[target_lang]
    )

    translation_log.append(translated)

    # Streaming TTS
    speech = generate_streaming_speech(translated)

    return "\n".join(transcript_log), "\n".join(translation_log), speech


def clear_conversation():

    global transcript_log
    global translation_log
    global audio_buffer

    transcript_log.clear()
    translation_log.clear()

    audio_buffer = None

    return "", "", None


# ==========================
# Gradio UI
# ==========================

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

    translated_audio = gr.Audio(
        label="Dubbed Translation"
    )

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
