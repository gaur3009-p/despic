from faster_whisper import WhisperModel
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


model = WhisperModel(
    "large-v3",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)


def transcribe_travel(audio_path):

    segments, info = model.transcribe(
        audio_path,

        # decoding
        beam_size=5,
        best_of=5,

        # robustness
        temperature=[0.0, 0.2, 0.4],

        # reduce hallucination
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,

        # speech specific
        vad_filter=True,

        # faster for short audio
        condition_on_previous_text=False,

        # better conversational context
        initial_prompt="This is a conversation."
    )

    text = ""

    for seg in segments:
        text += seg.text

    return text.strip(), info.language
