import numpy as np


class AudioBuffer:

    def __init__(self, sample_rate, buffer_seconds=1):

        self.sample_rate = sample_rate
        self.buffer_samples = sample_rate * buffer_seconds
        self.buffer = np.array([])

        self.MAX_BUFFER = sample_rate * 10

    def append(self, audio_chunk):

        if self.buffer.size == 0:
            self.buffer = audio_chunk
        else:
            self.buffer = np.concatenate([self.buffer, audio_chunk])

        if len(self.buffer) > self.MAX_BUFFER:
            self.buffer = self.buffer[-self.MAX_BUFFER:]

    def ready(self):

        return len(self.buffer) >= self.buffer_samples

    def get_buffer(self):

        return self.buffer

    def trim(self, end_index):

        self.buffer = self.buffer[end_index:]
