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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_play_sound_once(mumble_process: MumbleProcess):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Why is the sky blue?")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    mumble_process.on_play(audio)
    assert not mumble_process.is_playing()
    assert mumble_process.get_number_of_items_to_play()
    time.sleep(0.5)
    assert not mumble_process.get_number_of_items_to_play()
    assert mumble_process.is_playing()
    time.sleep(3)
    assert not mumble_process.is_playing()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_play_sound_twice(mumble_process: MumbleProcess):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)
    speech, samplerate  = tts.synthesize_stream("Why is the sky blue?")
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    mumble_process.on_play(audio)
    mumble_process.on_play(audio)

    assert not mumble_process.is_playing()
    assert mumble_process.get_number_of_items_to_play() > 1
    time.sleep(0.5)
    assert mumble_process.get_number_of_items_to_play() > 0
    assert mumble_process.is_playing()
    time.sleep(5)
    assert not mumble_process.is_playing()


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
    mumble_process.on_play(audio)
    assert not mumble_process.is_playing()
    assert mumble_process.get_number_of_items_to_play()
    time.sleep(0.5)
    assert not mumble_process.get_number_of_items_to_play()
    assert mumble_process.is_playing()

    event_bus.publish(TOPIC_MUMBLE_INTERRUPT_AUDIO, None)
    # assert mumble_process.is_interrupted()

    time.sleep(0.5)
    assert not mumble_process.is_interrupted()
    assert not mumble_process.is_playing()
    time.sleep(5)

