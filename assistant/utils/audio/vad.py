from typing import Callable
from pysilero_vad import SileroVoiceActivityDetector
import numpy as np
from collections import deque


class VadFilter:
    def __init__(
        self,
        callback: Callable,
        min_speech: int = 4,
        silence_end: int = 8,
        speech_threshold: float = 0.5,
        preroll_size: int = 5,
    ):
        self.vad = SileroVoiceActivityDetector()

        self.callback = callback

        self.min_speech = min_speech
        self.silence_end = silence_end
        self.speech_threshold = speech_threshold
        self.preroll_size = preroll_size

        self.speech_count = 0
        self.silence_count = 0
        self.speaking = False

        self.current_speech = bytearray()
        self.preroll_buffer = deque(maxlen=preroll_size)

    def __call__(self, chunk: np.ndarray) -> bool:
        self.preroll_buffer.append(chunk.copy())
        is_speech = self.vad(chunk.tobytes()) >= self.speech_threshold

        if is_speech:
            self.speech_count += 1
            self.silence_count = 0

            if self.speech_count == self.min_speech:
                self.speaking = True

                # First, add all the preroll chunks to the speech buffer
                for preroll_chunk in self.preroll_buffer:
                    self.current_speech.extend(preroll_chunk)

                # Then add the current chunk
                self.current_speech.extend(chunk)
            elif self.speaking:
                self.current_speech.extend(chunk)
        else:
            self.silence_count += 1

            if self.speaking:
                self.current_speech.extend(chunk)

                if self.silence_count >= self.silence_end:
                    if self.callback and callable(self.callback):
                        self.callback(bytes(self.current_speech))

                    self.speaking = False
                    self.speech_count = 0
                    self.silence_count = 0
                    self.current_speech = bytearray()
        return is_speech

