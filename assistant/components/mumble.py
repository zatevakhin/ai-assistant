import zenoh
from zenoh import Sample
import time
import logging
import numpy as np
from queue import Queue, Empty
from typing import Tuple
import threading
from pymumble_py3 import Mumble
from pymumble_py3.callbacks import PYMUMBLE_CLBK_SOUNDRECEIVED
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk

from .audio import AudioBufferTransformer, audio_length
from assistant.config import (
    ASSISTANT_NAME,
    MUMBLE_SERVER_HOST,
    MUMBLE_SERVER_PASSWORD,
    MUMBLE_SERVER_PORT,
    SPEECH_PIPELINE_SAMPLERATE,
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    TOPIC_MUMBLE_SOUND_NEW,
    TOPIC_MUMBLE_PLAY_AUDIO,
    TOPIC_MUMBLE_INTERRUPT_AUDIO,
)

logger = logging.getLogger(__name__)

class MumbleProcess:
    def __init__(self) -> None:
        self.ignored_users = [ASSISTANT_NAME]
        self.client = self.create_client()
        self.buffer_transformer = AudioBufferTransformer(
            self.on_new_sound, SPEECH_PIPELINE_SAMPLERATE, SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
        )

        self.running = False
        self.thread = threading.Thread(target=self._play_audio_in_mumble)
        self.is_playing = threading.Event()

        self.sound_queue: Queue[Tuple[int, np.ndarray[np.int16]]] = Queue()

        self.zenoh_session = zenoh.open(zenoh.Config())
        self.pub_mumble_sound = self.zenoh_session.declare_publisher(TOPIC_MUMBLE_SOUND_NEW)
        self.sub_play_audio = self.zenoh_session.declare_subscriber(TOPIC_MUMBLE_PLAY_AUDIO, self.on_play_audio)
        self.sub_on_interruption = self.zenoh_session.declare_subscriber(TOPIC_MUMBLE_INTERRUPT_AUDIO, self.on_interruption)
        logger.info("Mumble ... IDLE")

    @staticmethod
    def create_client():
        return Mumble(
            host=MUMBLE_SERVER_HOST,
            user=ASSISTANT_NAME,
            port=MUMBLE_SERVER_PORT,
            password=MUMBLE_SERVER_PASSWORD,
        )

    def on_interruption(self, sample: Sample):
        logger.warning(f"on_interruption({sample.payload})")

        if self.is_playing.is_set() and self.sound_queue.qsize() > 0:
            # NOTE: This will interrupt only on the end of sentence.
            self.sound_queue = Queue()

    def on_play_audio(self, sample: Sample):
        logger.info(f"> on_play_audio({type(sample)})")
        sound = np.frombuffer(sample.payload, dtype=np.int16)
        length = audio_length(sound, PYMUMBLE_SAMPLERATE)

        self.sound_queue.put((length, sound))

    def _play_audio_in_mumble(self):
        while self.running:
            try:
                length, sound = self.sound_queue.get(timeout=1)
                self.is_playing.set()
            except Empty:
                continue

            logger.info(f"Theres is '{self.sound_queue.qsize()}' audio chunks in queue.")
            logger.info(f"Current chunk is {length}ms long.")
            # NOTE: Audio chunks can be even smaller to have faster interruption.
            self.client.sound_output.add_sound(sound.tobytes())
            time.sleep(length / 1000)
            self.is_playing.clear()


    def on_new_sound(self, sound: np.ndarray[np.float32]):
        logger.debug(f"{type(sound)}, {sound.flags.writeable}")
        self.pub_mumble_sound.put(sound.tobytes())

    def sound_received_callback(self, user: str, soundchunk: SoundChunk):
        if user in self.ignored_users:
            logger.info(f"Sound from '{user}' was ignored. Because of ignore list.")
            return

        self.buffer_transformer(soundchunk)

    def run(self):
        self.running = True
        self.client.callbacks.set_callback(PYMUMBLE_CLBK_SOUNDRECEIVED, self.sound_received_callback)
        self.client.set_receive_sound(True)
        self.client.start()
        self.thread.start()
        time.sleep(3)
        logger.info("Mumble ... OK")

    def stop(self):
        logger.info("Mumble ... STOPPING")
        self.running = False
        self.client.stop()
        self.pub_mumble_sound.undeclare()
        self.zenoh_session.close()
        self.thread.join()
        logger.info("Mumble ... DEAD")

def main():
    pass

if __name__ == "__main__":
    main()
