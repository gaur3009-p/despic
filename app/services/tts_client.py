import requests

TTS_URL = "http://127.0.0.1:8001/generate"

def speak(text):

    response = requests.post(
        TTS_URL,
        json={"text": text}
    )

    data = response.json()

    return data["audio_file"]
