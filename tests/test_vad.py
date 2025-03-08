from queue import Queue
from time import sleep
from numpy.typing import NDArray
import pytest
import resampy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistant.components import VadProcess, EventBus, EventType
from assistant.config import PIPER_TTS_MODEL, PIPER_MODELS_LOCATION, SPEECH_PIPELINE_BUFFER_SIZE_MILIS
from voice_forge import PiperTts
from assistant.components.audio import chop_audio, enrich_with_silence
import numpy as np

@pytest.fixture(scope="module")
def event_bus():
    eb = EventBus()
    yield eb

@pytest.fixture(scope="module")
def vad_process(event_bus: EventBus):
    process = VadProcess(event_bus)

    process.run()
    yield process
    process.stop()


def test_is_speech(_: VadProcess, event_bus: EventBus):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Hello, World!")
    audio = resampy.resample(speech.astype(np.float32) / 32768.0, samplerate, 16000)
    audio = enrich_with_silence(audio, 16000, 0.2, 1.0)

    speech_queue: Queue[NDArray[np.float32]] = Queue()

    def on_speech(seg: NDArray[np.float32]):
        speech_queue.put(seg)

    event_bus.subscribe(EventType.VAD_NEW_SPEECH, on_speech)

    for seg in chop_audio(audio, 16000, SPEECH_PIPELINE_BUFFER_SIZE_MILIS):
        event_bus.publish(EventType.MUMBLE_NEW_AUDIO, seg)
        sleep(SPEECH_PIPELINE_BUFFER_SIZE_MILIS / 1000.0)

    sleep(3)

    assert speech_queue.qsize() > 0


