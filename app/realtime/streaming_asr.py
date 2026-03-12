from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    "medium",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)


class StreamingASR:

    def __init__(self):

        self.last_text = ""

    def transcribe(self, audio_path):

        segments, info = model.transcribe(
            audio_path,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False
        )

        text = ""

        for segment in segments:
            text += segment.text

        text = text.strip()

        if text == self.last_text:
            return None, info.language

        self.last_text = text

        return text, info.language
