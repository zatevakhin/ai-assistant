import os
from datetime import datetime
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import resampy
import soundfile as sf
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk


class AudioBufferTransformer:
    def __init__(
        self,
        callback: Callable[[NDArray[np.float32]], None],
        target_samplerate: int,
        target_chunk_length_ms: int,
    ):
        """
        Initialize the AudioBufferHandler.

        :param callback: Callback where processed audio chunks will be forwarded.
        :param target_samplerate: The desired sample rate for the output audio.
        :param target_chunk_length_ms: The desired length of audio chunks in milliseconds.
        """
        self.callback = callback
        self.target_samplerate = target_samplerate
        self.target_chunk_length_ms = target_chunk_length_ms
        self.audio_buffer = np.array([], dtype=np.int16)
        self.input_samplerate = PYMUMBLE_SAMPLERATE

    def __call__(self, soundchunk: SoundChunk):
        """
        Callable method to handle incoming sound chunks.

        :param soundchunk: The incoming sound chunk.
        """
        # Process the incoming sound chunk
        self._process_chunk(soundchunk)

    def _process_chunk(self, soundchunk: SoundChunk):
        """
        Process the incoming sound chunk.

        :param soundchunk: The incoming sound chunk.
        """
        # Convert byte data to numpy array
        numpy_array = np.frombuffer(soundchunk.pcm, dtype=np.int16)

        # Append this chunk to the buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, numpy_array))

        # Check if the buffer has reached the desired length
        if self._buffer_length_ms() >= self.target_chunk_length_ms:
            self._process_buffer()

    def _buffer_length_ms(self) -> float:
        """
        Calculate the length of the current buffer in milliseconds.

        :return: Length of the buffer in milliseconds.
        """
        return len(self.audio_buffer) / self.input_samplerate * 1000

    def _process_buffer(self):
        """
        Process the audio buffer.
        """
        # Calculate the number of samples corresponding to the target chunk length
        target_samples = int(self.target_chunk_length_ms / 1000 * self.input_samplerate)

        # Take only the necessary part of the buffer for processing
        buffer_to_process = self.audio_buffer[:target_samples]

        # Resample the buffer
        audio = buffer_to_process.astype(np.float32, order="C") / np.float32(
            np.iinfo(buffer_to_process.dtype).max
        )
        resampled = resampy.resample(
            audio, self.input_samplerate, self.target_samplerate
        )

        # Put the processed buffer in the output callback
        self.callback(resampled)

        # Retain the excess part of the buffer for the next chunk
        self.audio_buffer = self.audio_buffer[target_samples:]


def record_audio_chunk(
    chunk: NDArray[np.float32],
    directory: str,
    samplerate: int,
    audio_format: str = "FLAC",
) -> None:
    # Validate audio format
    if audio_format not in ["WAV", "FLAC"]:
        raise ValueError("Unsupported audio format. Please use 'WAV' or 'FLAC'.")

    # Use provided timestamp or generate a new one
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{timestamp}.{audio_format.lower()}")

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Append the audio chunk to the file
    mode = "w" if not os.path.exists(filename) else "a"
    with sf.SoundFile(
        filename, mode=mode, samplerate=samplerate, channels=1, format=audio_format
    ) as file:
        file.write(chunk)


def audio_length(audio_data: np.ndarray, sample_rate: int) -> int:
    """
    Calculate the length of the audio in milliseconds.

    :param audio_data: np.ndarray containing audio data
    :param sample_rate: Sample rate of the audio in Hz
    :return: Length of the audio in milliseconds
    """
    # Number of samples in the audio data
    num_samples = audio_data.shape[0]

    # Calculate length in seconds and convert to milliseconds
    length_in_seconds = num_samples / sample_rate
    length_in_milliseconds = int(length_in_seconds * 1000)

    return length_in_milliseconds


def chop_audio(
    audio_data: np.ndarray, sample_rate: int, segment_length_ms: int
) -> list:
    """
    Chop audio data into segments of given length.

    :param audio_data: np.ndarray containing audio data
    :param sample_rate: Sample rate of the audio in Hz
    :param segment_length_ms: Length of each segment in milliseconds
    :return: List of numpy arrays, each representing a segment of the audio
    """
    # Calculate number of samples in each segment
    num_samples_per_segment = int(sample_rate * segment_length_ms / 1000)

    # Calculate total number of segments
    total_segments = len(audio_data) // num_samples_per_segment

    # Chop audio into segments
    segments = [
        audio_data[i * num_samples_per_segment : (i + 1) * num_samples_per_segment]
        for i in range(total_segments)
    ]

    return segments


def enrich_with_silence(
    audio: np.ndarray, samplerate: int, start_sec: float, end_sec: float
) -> np.ndarray:
    end = np.zeros(int(samplerate * end_sec), dtype=audio.dtype)
    start = np.zeros(int(samplerate * start_sec), dtype=audio.dtype)
    return np.concatenate((start, audio, end))
