import time
import warnings
from queue import Queue

import numpy as np
import pytest
import resampy

from assistant.components.audio import audio_length

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pymumble_py3
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from voice_forge import PiperTts

from assistant.components import EventBus, EventType, MumbleProcess, Sentence
from assistant.config import PIPER_MODELS_LOCATION, PIPER_TTS_MODEL


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
    text = "Why is the sky blue?"
    speech, samplerate = tts.synthesize_stream(text)
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    sentence = Sentence(
        text=text, audio=audio, length=audio_length(audio, PYMUMBLE_SAMPLERATE)
    )
    mumble_process.on_play(sentence)
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
    text = "Why is the sky blue?"
    speech, samplerate = tts.synthesize_stream(text)
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    sentence = Sentence(
        text=text, audio=audio, length=audio_length(audio, PYMUMBLE_SAMPLERATE)
    )

    mumble_process.on_play(sentence)
    mumble_process.on_play(sentence)

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
    text = "Test, Test, Test!"
    speech, samplerate = tts.synthesize_stream(text)
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    sentence = Sentence(
        text=text, audio=audio, length=audio_length(audio, PYMUMBLE_SAMPLERATE)
    )

    sound_chunks = Queue()

    def on_sound(data):
        sound_chunks.put(data)

    event_bus.subscribe(EventType.MUMBLE_NEW_AUDIO, on_sound)

    client = pymumble_py3.Mumble(host="localhost", user="pytest")
    client.start()
    client.is_ready()
    client.sound_output.add_sound(sentence.audio.tobytes())
    time.sleep(3)
    client.stop()

    assert sound_chunks.qsize() > 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_play_sound_with_interrupt(mumble_process: MumbleProcess, event_bus: EventBus):
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)

    text = "Try to interrupt me!"
    speech, samplerate = tts.synthesize_stream(text)
    audio = resampy.resample(speech, samplerate, PYMUMBLE_SAMPLERATE).astype(np.int16)
    sentence = Sentence(
        text=text, audio=audio, length=audio_length(audio, PYMUMBLE_SAMPLERATE)
    )

    mumble_process.on_play(sentence)

    assert not mumble_process.is_playing()
    assert mumble_process.get_number_of_items_to_play()
    time.sleep(0.5)
    assert not mumble_process.get_number_of_items_to_play()
    assert mumble_process.is_playing()

    event_bus.publish(EventType.MUMBLE_INTERRUPT_AUDIO, None)

    time.sleep(0.5)
    assert not mumble_process.is_interrupted()
    assert not mumble_process.is_playing()
    time.sleep(5)
