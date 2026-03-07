import requests

TTS_URL = "http://localhost:8001/generate"

def speak(text):

    response = requests.post(
        TTS_URL,
        json={"text": text}
    )

    data = response.json()

    return data["audio_file"]
