import numpy as np
from .vad import VadFilter


def audio_length(audio_data: np.ndarray, samplerate: int) -> int:
    num_samples = audio_data.shape[0]
    return int((num_samples / samplerate) * 1000)


def chop_audio(
    audio_data: np.ndarray, samplerate: int, segment_length_ms: int
) -> list:
    num_samples_per_segment = int(samplerate * segment_length_ms / 1000)

    total_segments = len(audio_data) // num_samples_per_segment

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
