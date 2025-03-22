import logging
from assistant.core import Plugin, service
from typing import List
from numpy.typing import NDArray
import numpy as np
import threading
from queue import Queue
import reactivex as rx
from reactivex import operators as ops

from pymumble_py3 import Mumble
from pymumble_py3.callbacks import (
    PYMUMBLE_CLBK_SOUNDRECEIVED,
    PYMUMBLE_CLBK_CONNECTED,
    PYMUMBLE_CLBK_DISCONNECTED,
)
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from assistant.config import (
    ASSISTANT_NAME,
    MUMBLE_SERVER_HOST,
    MUMBLE_SERVER_PASSWORD,
    MUMBLE_SERVER_PORT,
    MUMBLE_SERVER_CHANNEL,
    SPEECH_PIPELINE_SAMPLERATE,
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
)

from assistant.components.util import observe
from assistant.components.audio import AudioBufferTransformer, chop_audio
from assistant.components.synthesis import Sentence
from . import events


class MumbleInterface(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return [
            events.MUMBLE_CLIENT_CONNECTED,
            events.MUMBLE_CLIENT_DISCONNECTED,
            events.MUMBLE_AUDIO_CHUNK,
            events.MUMBLE_AUDIO_PLAY,
            events.MUMBLE_PLAYBACK_DONE,
            events.MUMBLE_PLAYBACK_IN_PROGRESS,
            events.MUMBLE_PLAYBACK_INTERRUPT,
        ]

    def initialize(self) -> None:
        super().initialize()

        # TODO: Get or create
        self.client = Mumble(
            host=MUMBLE_SERVER_HOST,
            user=ASSISTANT_NAME,
            port=MUMBLE_SERVER_PORT,
            password=MUMBLE_SERVER_PASSWORD,
        )

        self.playback_queue = Queue()
        sub = observe(self.playback_queue, self.on_play_from_queue)
        self.subscriptions.append(sub)

        self.is_connected = threading.Event()
        self.is_interrupted = threading.Event()
        self.is_playback_done = threading.Event()
        self.is_playback_in_progress = threading.Event()

        self.buffer_transformer = AudioBufferTransformer(
            self.on_sound,
            SPEECH_PIPELINE_SAMPLERATE,
            SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
        )

        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED, lambda _, chunk: self.buffer_transformer(chunk)
        )
        self.client.callbacks.set_callback(PYMUMBLE_CLBK_CONNECTED, self.on_connect)
        self.client.callbacks.set_callback(
            PYMUMBLE_CLBK_DISCONNECTED, self.on_disconnect
        )
        self.client.set_receive_sound(True)
        self.client.start()
        self.logger.info(
            f"Plugin '{self.name}' connecting to server: '{MUMBLE_SERVER_HOST}:{MUMBLE_SERVER_PORT}'"
        )
        self.client.is_ready() # waits connection

        if MUMBLE_SERVER_CHANNEL:
            if channel := self.client.channels.find_by_name(MUMBLE_SERVER_CHANNEL):
                channel.move_in()

        self.event_bus.subscribe(events.MUMBLE_AUDIO_PLAY, self.on_play)
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' disconnection from server.")
        self.client.stop()

    def on_connect(self):
        self.logger.info(f"Plugin '{self.name}' connected")
        self.is_connected.set()
        self.event_bus.publish(events.MUMBLE_CLIENT_CONNECTED, None)

    def on_disconnect(self):
        self.logger.info(f"Plugin '{self.name}' disconnected")
        self.is_connected.clear()
        self.event_bus.publish(events.MUMBLE_CLIENT_DISCONNECTED, None)

    def on_sound(self, sound: NDArray[np.float32]):
        self.logger.debug(f"{type(sound)}, {sound.flags.writeable}")
        self.event_bus.publish(events.MUMBLE_AUDIO_CHUNK, sound)

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
                self.event_bus.publish(events.MUMBLE_PLAYBACK_INTERRUPT, None)
                while not self.playback_queue.empty():
                    self.playback_queue.get()
                    self.playback_queue.task_done()

                self.is_interrupted.clear()
                self.is_playback_in_progress.clear()
                self.is_playback_done.set()

        def on_playback_complete():
            self.is_playback_in_progress.clear()
            self.is_playback_done.set()
            self.event_bus.publish(events.MUMBLE_PLAYBACK_DONE, None)

        _ = rx.zip(
            rx.interval(0.020),
            rx.from_iterable(chop_audio(sentence.audio, PYMUMBLE_SAMPLERATE, 20)),
        ).pipe(
            ops.map(lambda x: x[1].tobytes()),
            ops.do_action(self.client.sound_output.add_sound),
            ops.finally_action(on_interrupt),
        ).subscribe(on_completed=on_playback_complete)

        self.event_bus.publish(events.MUMBLE_PLAYBACK_IN_PROGRESS, sentence)
        self.is_playback_done.wait()
