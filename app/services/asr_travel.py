from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    "medium",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)


def transcribe_travel(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=3,
        best_of=3
    )

    text = ""

    for seg in segments:
        text += seg.text

    return text.strip(), info.language
