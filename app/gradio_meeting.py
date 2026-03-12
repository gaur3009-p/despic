import gradio as gr
import tempfile
import soundfile as sf
import librosa

from realtime.vad import get_speech_segments
from realtime.audio_buffer import AudioBuffer
from realtime.streaming_asr import StreamingASR
from services.streaming_translate import translate_chunk

from services.streaming_tts import generate_streaming_speech


asr_engine = StreamingASR()

audio_buffer = None

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

    # resample for VAD + Whisper
    if sr != 16000:
        data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
        sr = 16000

    if audio_buffer is None:
        audio_buffer = AudioBuffer(sr, buffer_seconds=1)

    audio_buffer.append(data)

    if not audio_buffer.ready():
        return "\n".join(transcript_log), "\n".join(translation_log), None

    buffer_audio = audio_buffer.get_buffer()

    speech_segments = get_speech_segments(buffer_audio, sr)

    if len(speech_segments) == 0:
        return "\n".join(transcript_log), "\n".join(translation_log), None

    segment = speech_segments[0]

    speech_audio = buffer_audio[segment["start"]:segment["end"]]

    audio_buffer.trim(segment["end"])

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, speech_audio, sr)

    text, source_lang = asr_engine.transcribe(temp.name)

    if text is None or text.strip() == "":
        return "\n".join(transcript_log), "\n".join(translation_log), None

    transcript_log.append(text)

    translated = translate_chunk(
        text,
        source_lang,
        languages[target_lang]
    )

    translation_log.append(translated)

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
        type="numpy"
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

    translated_audio = gr.Audio()

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
