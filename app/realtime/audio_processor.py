from services.asr import transcribe
from services.translate import translate
from services.tts import generate_speech


def process_audio(audio_path, target_language):

    text = transcribe(audio_path)

    translated = translate(text, target_language)

    audio_file = generate_speech(translated)

    return audio_file
