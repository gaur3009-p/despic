import torch
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()


def get_speech_segments(audio, sample_rate):

    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate
    )

    return speech_timestamps
