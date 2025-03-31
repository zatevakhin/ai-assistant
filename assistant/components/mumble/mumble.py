import threading
from datetime import datetime
from functools import partial
from queue import Queue
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np
import reactivex as rx
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from pymumble_py3 import Mumble
from pymumble_py3.callbacks import (
    PYMUMBLE_CLBK_CONNECTED,
    PYMUMBLE_CLBK_DISCONNECTED,
    PYMUMBLE_CLBK_SOUNDRECEIVED,
    PYMUMBLE_CLBK_USERUPDATED,
)
from pymumble_py3.channels import Channel
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk
from pymumble_py3.users import User
from reactivex import operators as ops

from assistant.config import (
    ASSISTANT_NAME,
    SPEECH_PIPELINE_SAMPLERATE,
)
from assistant.core import service
from assistant.utils import observe
from assistant.core.component import Component
from assistant.utils.audio import VadFilter
from assistant.utils.audio.reshape import FixedLengthAudioChunker
from . import events


class Sentence(BaseModel):
    text: str
    audio: Any
    length: float


class SpeechSegment(BaseModel):
    source: str
    data: NDArray[np.int16] = Field(repr=False)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class MumbleInterface(Component):
    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return [
            events.MUMBLE_CLIENT_CONNECTED,
            events.MUMBLE_CLIENT_DISCONNECTED,
            events.MUMBLE_AUDIO_CHUNK,
            events.MUMBLE_AUDIO_SPEECH,
            events.MUMBLE_AUDIO_PLAY,
            events.MUMBLE_PLAYBACK_DONE,
            events.MUMBLE_PLAYBACK_IN_PROGRESS,
            events.MUMBLE_PLAYBACK_INTERRUPT,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.logger.setLevel(self.get_config("log_level", "DEBUG"))
        mumble_server = self.get_config("server", {})
        mumble_host = mumble_server.get("host", "127.0.0.1")
        mumble_port = mumble_server.get("port", 64738)
        mumble_password = mumble_server.get("password", "")
        mumble_channel = mumble_server.get("channel", None)

        # TODO: Get or create
        self.client = Mumble(
            host=mumble_host,
            port=mumble_port,
            password=mumble_password,
            user=ASSISTANT_NAME,
        )

        self.playback_queue = Queue()
        sub = observe(self.playback_queue, self.on_play_from_queue)

        self.is_interrupted = threading.Event()
        self.is_playback_done = threading.Event()
        self.is_playback_in_progress = threading.Event()

        self.fixed_chunker_for_source: Dict[str, FixedLengthAudioChunker] = {}
        self.speech_filter_for_source: Dict[str, VadFilter] = {}

        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED, self.on_sound_from_source
        )
        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_CONNECTED,
            self.proxy(events.MUMBLE_CLIENT_CONNECTED),
        )
        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_USERUPDATED, self.on_user_updated
        )

        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_DISCONNECTED,
            self.proxy(events.MUMBLE_CLIENT_DISCONNECTED),
        )
        self.client.set_receive_sound(True)
        self.client.start()
        self.logger.info(
            f"Plugin '{self.name}' connecting to server: '{mumble_host}:{mumble_port}'"
        )
        self.client.is_ready()  # waits connection

        if mumble_channel:
            if channel := self.client.channels.find_by_name(mumble_channel):
                channel: Channel = channel
                channel.move_in()
                # NOTE: Need to wait a bit before getting list of users on channel.
                sleep(1)

        for session in self.client.my_channel().get_users():
            user: User = self.client.users[session["session"]]
            self.add_speech_filter(user)

        # self.event_bus.subscribe(events.MUMBLE_AUDIO_PLAY, self.on_play)
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def add_speech_filter(self, source: User):
        username = source.get_property("name")
        assert username is not None

        if username == ASSISTANT_NAME:
            self.logger.info(f"Ignored source '{username}', because its assistant.")
            return False

        if username not in self.speech_filter_for_source:
            self.speech_filter_for_source[username] = VadFilter(
                partial(self.on_speech, source),
            )

        if username not in self.fixed_chunker_for_source:
            self.fixed_chunker_for_source[username] = FixedLengthAudioChunker(
                callback=lambda chunk: self.speech_filter_for_source[username](chunk),
                target_chunk_length_ms=32,
                source_samplerate=PYMUMBLE_SAMPLERATE,
                target_samplerate=SPEECH_PIPELINE_SAMPLERATE,
            )

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' disconnection from server.")
        self.client.stop()

    def on_user_updated(self, session, attributes):
        self.logger.info(f"on_user_updated({session}, {attributes})")
        my_channel = self.client.my_channel().get("channel_id")

        if my_channel == session.get("channel_id"):
            user: User = self.client.users[session["session"]]
            self.add_speech_filter(user)

    def on_sound_from_source(self, source: dict, chunk: SoundChunk):
        username = source.get("name", None)
        assert username is not None
        self.fixed_chunker_for_source[username](chunk.pcm)

    def on_speech(self, user: User, speech: bytes):
        self.logger.info(f"{type(speech)}, {user}")
        username = str(user.get_property("name"))

        buffer = np.frombuffer(speech, dtype=np.int16)
        segment = SpeechSegment(source=username, data=buffer)

        self.proxy(events.MUMBLE_AUDIO_SPEECH)(segment)

    def on_play(self, sentence: Sentence):
        self.logger.info(f"> on_play('{sentence.text}')")
        self.playback_queue.put(sentence)

    @service
    async def play_audio(self, sentence: Sentence):
        pass

    def on_play_from_queue(self, sentence: Sentence):
        self.is_playback_done.clear()
        self.is_playback_in_progress.set()

        def on_interrupt():
            if self.is_interrupted.is_set():
                self.proxy(events.MUMBLE_PLAYBACK_INTERRUPT)()
                while not self.playback_queue.empty():
                    self.playback_queue.get()
                    self.playback_queue.task_done()

                self.is_interrupted.clear()
                self.is_playback_in_progress.clear()
                self.is_playback_done.set()

        def on_playback_complete():
            self.is_playback_in_progress.clear()
            self.is_playback_done.set()
            self.proxy(events.MUMBLE_PLAYBACK_DONE)()

        _ = (
            rx.zip(
                rx.interval(0.020),
                rx.from_iterable(chop_audio(sentence.audio, PYMUMBLE_SAMPLERATE, 20)),
            )
            .pipe(
                ops.map(lambda x: x[1].tobytes()),
                ops.do_action(self.client.sound_output.add_sound),
                ops.finally_action(on_interrupt),
            )
            .subscribe(on_completed=on_playback_complete)
        )

        # self.event_bus.publish(events.MUMBLE_PLAYBACK_IN_PROGRESS, sentence)
        self.is_playback_done.wait()
