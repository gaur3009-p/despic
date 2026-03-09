from fastapi import FastAPI
from pydantic import BaseModel
from TTS.api import TTS
import uuid

app = FastAPI()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

class TTSRequest(BaseModel):
    text: str

@app.post("/generate")

def generate_speech(req: TTSRequest):

    file = f"/content/despic/tts_service/output_{uuid.uuid4()}.wav"

    tts.tts_to_file(
        text=req.text,
        file_path=file,
        language="hi"
    )

    return {"audio_file": file}
