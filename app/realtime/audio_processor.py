from services.asr import transcribe
from services.translate import translate
from services.tts import generate_speech

def process_audio(audio_file, target_language):

    text = transcribe(audio_file)

    translated = translate(text, target_language)

    speech = generate_speech(translated)

    return speech
