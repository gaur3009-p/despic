from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    "medium",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

def transcribe(audio_path):

    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    text = " ".join(segment.text for segment in segments)

    return text.strip()
