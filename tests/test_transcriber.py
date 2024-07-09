from queue import Queue
import pytest
import resampy
import warnings
from functools import partial
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistant.components import SpeechTranscriberProcess, EventBus, EventType
from assistant.config import PIPER_TTS_MODEL, PIPER_MODELS_LOCATION
from voice_forge import PiperTts
from voice_pulse import SpeechSegment
from assistant.components.audio import enrich_with_silence
import numpy as np
from datetime import datetime

@pytest.fixture(scope="module")
def event_bus():
    eb = EventBus()
    yield eb

@pytest.fixture(scope="module")
def transcriber(event_bus: EventBus):
    process = SpeechTranscriberProcess(event_bus)

    process.run()
    yield process
    process.stop()


def test_transcribe_speech(transcriber: SpeechTranscriberProcess, event_bus: EventBus):

    test_text = "Hello, World!"

    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream(test_text)
    audio = resampy.resample(speech.astype(np.float32) / 32768.0, samplerate, 16000)
    audio = enrich_with_silence(audio, 16000, 0.2, 1.0)

    transcriptions_queue = Queue()

    seg = SpeechSegment(speech=audio, timestamp=datetime.now())
    sub = event_bus.subscribe(EventType.TRANSCRIPTION_DONE, transcriptions_queue.put)
    event_bus.publish(EventType.VAD_NEW_SPEECH, seg)

    transcription = transcriptions_queue.get()

    strip_special_chars = partial(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x.lower()))
    assert strip_special_chars(test_text) == strip_special_chars(transcription["text"])

