import tempfile
import soundfile as sf
import librosa
import numpy as np

from realtime.vad import detect_speech
from services.asr_travel import transcribe_travel
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


def run_travel_pipeline(audio, sr, target_lang):

    # -------------------------
    # Convert stereo → mono
    # -------------------------
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # -------------------------
    # Ensure float32
    # -------------------------
    audio = audio.astype(np.float32)

    # -------------------------
    # Clip audio range
    # -------------------------
    audio = np.clip(audio, -1.0, 1.0)

    # -------------------------
    # Resample to 16k
    # -------------------------
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # -------------------------
    # Minimum audio length
    # -------------------------
    if len(audio) < 1600:  # ~0.1 sec
        return "", "", None

    # -------------------------
    # Voice Activity Detection
    # -------------------------
    speech_segments = detect_speech(audio, sr)

    if not speech_segments:
        return "", "", None

    # -------------------------
    # Take first speech segment
    # -------------------------
    segment = speech_segments[0]

    speech_audio = audio[
        segment["start"]:segment["end"]
    ]

    # -------------------------
    # Save temp audio
    # -------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:

        sf.write(temp.name, speech_audio, sr)

        temp_path = temp.name

    # -------------------------
    # Whisper transcription
    # -------------------------
    text, src_lang = transcribe_travel(temp_path)

    if text.strip() == "":
        return "", "", None

    # -------------------------
    # Translation
    # -------------------------
    translated = translate(
        text,
        src_lang,
        languages[target_lang]
    )

    # -------------------------
    # TTS
    # -------------------------
    speech = speak(translated)

    return text, translated, speech
