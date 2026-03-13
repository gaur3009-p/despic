from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()


def detect_speech(audio, sample_rate):

    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate
    )

    return timestamps
