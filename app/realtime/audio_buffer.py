import numpy as np


class AudioBuffer:

    def __init__(self, sr, seconds=3):

        self.sr = sr
        self.max_samples = sr * seconds
        self.buffer = np.array([])

    def append(self, data):

        if self.buffer.size == 0:
            self.buffer = data
        else:
            self.buffer = np.concatenate([self.buffer, data])

        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]

    def get(self):

        return self.buffer

    def clear(self):

        self.buffer = np.array([])
