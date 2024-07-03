import warnings
import threading
import pytest
import numpy as np
import time
import resampy
from queue import Queue

warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistant.components import MumbleProcess, EventBus
from assistant.config import PIPER_TTS_MODEL, PIPER_MODELS_LOCATION, TOPIC_MUMBLE_SOUND_NEW, TOPIC_MUMBLE_INTERRUPT_AUDIO
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from voice_forge import PiperTts
import pymumble_py3


@pytest.fixture(scope="module")
def event_bus():
    eb = EventBus()
    yield eb


@pytest.fixture(scope="module")
def mumble_process(event_bus):
    process = MumbleProcess(event_bus)
    process.run()
    yield process
    process.stop()

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_server_is_up(mumble_process: MumbleProcess):
    assert mumble_process.is_alive()

def test_connected_to_server(mumble_process: MumbleProcess):
    assert mumble_process.is_connected()

def test_play_sound(mumble_process: MumbleProcess):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Why is the sky blue?")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    obs = mumble_process.on_play(audio)

    assert obs.run()

def test_play_sound_twice(mumble_process: MumbleProcess):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Why is the sky blue?")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    obs1 = mumble_process.on_play(audio)
    obs2 = mumble_process.on_play(audio)

    assert obs1.run()
    assert obs2.run()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_receive_sound(event_bus: EventBus):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Test, Test, Test!")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)

    sound_chunks = Queue()

    def on_sound(data):
        sound_chunks.put(data)

    event_bus.subscribe(TOPIC_MUMBLE_SOUND_NEW, on_sound)

    client = pymumble_py3.Mumble(host="localhost", user="pytest")
    client.start()
    client.is_ready()
    client.sound_output.add_sound(audio.tobytes())
    time.sleep(3)
    client.stop()

    assert sound_chunks.qsize() > 0

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_play_sound_with_interrupt(mumble_process: MumbleProcess, event_bus: EventBus):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Try to interrupt me!")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    obs = mumble_process.on_play(audio)
    time.sleep(0.5)
    assert mumble_process.is_playing.is_set()
    assert not mumble_process.is_interrupted.is_set()

    event_bus.publish(TOPIC_MUMBLE_INTERRUPT_AUDIO, None)
    assert mumble_process.is_interrupted.is_set()

    time.sleep(0.5)
    assert not mumble_process.is_interrupted.is_set()
    assert not mumble_process.is_playing.is_set()
    assert obs.run()

