import tempfile
import soundfile as sf

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


def run_pipeline(audio, sr, target_lang):

    speech_segments = detect_speech(audio, sr)

    if len(speech_segments) == 0:
        return "", "", None

    segment = speech_segments[0]

    speech_audio = audio[segment["start"]:segment["end"]]

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp.name, speech_audio, sr)

    text, src_lang = transcribe(temp.name)

    if text == "":
        return "", "", None

    translated = translate(text, src_lang, languages[target_lang])

    speech = speak(translated)

    return text, translated, speech
