import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()


def detect_speech(audio, sample_rate):

    # convert numpy → float32
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)

    # convert → torch tensor
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    # reset model state
    model.reset_states()

    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate
    )

    return timestamps
