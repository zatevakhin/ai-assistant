from voice_pulse import Listener, Config, VadEngine
from voice_pulse.input_sources import CallbackInput
import numpy as np
import threading
from zenoh import Sample
import zenoh
import logging


from assistant.config import (
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    VAD_SILENCE_THRESHOLD,
    TOPIC_MUMBLE_SOUND_NEW,
    TOPIC_VAD_SPEECH_NEW
)

logger = logging.getLogger(__name__)

class VadProcess:
    def __init__(self):
        self.config = Config(
            vad_engine=VadEngine.SILERIO,
            block_duration=SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
            silence_threshold=VAD_SILENCE_THRESHOLD,
        )
        self.stream = CallbackInput(self.config.blocksize)
        self.thread = threading.Thread(target=self._process_speech)
        self.running = False

        self.zenoh_session = zenoh.open(zenoh.Config())
        self.sub_mumble_sound = self.zenoh_session.declare_subscriber(TOPIC_MUMBLE_SOUND_NEW, self.on_new_sound)
        self.pub_new_speech = self.zenoh_session.declare_publisher(TOPIC_VAD_SPEECH_NEW)

        logger.info("VAD Process ... IDLE")

    def on_new_sound(self, sample: Sample):
        logger.debug(f"> on_new_sound({type(sample)})")
        sound = np.frombuffer(sample.payload, dtype=np.float32)
        sound = np.array(sound) # NOTE: Convert to Writable numpy array.
        self.stream.receive_chunk(sound)

    def _process_speech(self):
        for speech in Listener(self.config, self.stream):
            if not self.running:
                break
            self.pub_new_speech.put(speech.tobytes())

    def run(self):
        self.running = True
        self.thread.start()
        logger.info("VAD Process ... OK")

    def stop(self):
        logger.info("VAD Process ... STOPPING")
        self.running = False
        self.sub_mumble_sound.undeclare()
        self.zenoh_session.close()
        self.stream.receive_chunk(None)
        self.thread.join()
        logger.info("VAD Process ... DEAD")
