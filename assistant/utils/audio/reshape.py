from typing import Callable
import numpy as np
import resampy
from numpy.typing import NDArray


class FixedLengthAudioChunker:
    def __init__(
        self,
        callback: Callable[[NDArray[np.int16]], None],
        target_samplerate: int,
        source_samplerate: int,
        target_chunk_length_ms: int,
    ):
        self.callback = callback
        self.target_samplerate = target_samplerate
        self.target_chunk_length_ms = target_chunk_length_ms
        self.audio_buffer = np.array([], dtype=np.int16)
        self.input_samplerate = source_samplerate

    def __call__(self, chunk: bytes):
        self._process_chunk(chunk)

    def _process_chunk(self, chunk: bytes):
        numpy_array = np.frombuffer(chunk, dtype=np.int16)

        self.audio_buffer = np.concatenate((self.audio_buffer, numpy_array))

        if self._buffer_length_ms() >= self.target_chunk_length_ms:
            self._process_buffer()

    def _buffer_length_ms(self) -> float:
        return len(self.audio_buffer) / self.input_samplerate * 1000

    def _process_buffer(self):
        target_samples = int(self.target_chunk_length_ms / 1000 * self.input_samplerate)

        buffer_to_process = self.audio_buffer[:target_samples]

        # Normalize to float32 in [-1.0, 1.0] range
        audio = buffer_to_process.astype(np.float32, order="C") / np.float32(
            np.iinfo(buffer_to_process.dtype).max
        )

        # Resample (output is still in [-1.0, 1.0] range)
        resampled_float = resampy.resample(
            audio, self.input_samplerate, self.target_samplerate
        )

        # Scale back to int16 range and convert
        resampled = (resampled_float * np.iinfo(np.int16).max).astype(np.int16)

        self.callback(resampled)

        self.audio_buffer = self.audio_buffer[target_samples:]

