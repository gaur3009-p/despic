import tempfile
import soundfile as sf
import librosa
import numpy as np

from realtime.vad import detect_speech
from services.asr import transcribe
from services.translator import translate
from services.tts_engine import speak


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


# global buffers
audio_buffer = np.array([], dtype=np.float32)
transcript_buffer = ""


def run_pipeline(audio, sr, target_lang):

    global audio_buffer
    global transcript_buffer

    if audio is None:
        return transcript_buffer, "", None

    # stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    # resample
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # accumulate audio
    audio_buffer = np.concatenate([audio_buffer, audio])

    # if buffer too small
    if len(audio_buffer) < 16000:
        return transcript_buffer, "", None

    speech_segments = detect_speech(audio_buffer, sr)

    if not speech_segments:
        return transcript_buffer, "", None

    segment = speech_segments[-1]

    speech_audio = audio_buffer[
        segment["start"]:segment["end"]
    ]

    # clear processed buffer
    audio_buffer = audio_buffer[segment["end"]:]

    # save chunk
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, speech_audio, sr)

    text, src_lang = transcribe(temp.name)

    if text.strip() == "":
        return transcript_buffer, "", None

    transcript_buffer += " " + text

    translated = translate(
        transcript_buffer,
        src_lang,
        languages[target_lang]
    )

    speech = speak(translated)

    return transcript_buffer.strip(), translated, speech
