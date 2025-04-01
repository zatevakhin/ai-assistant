from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, Optional
import os
import numpy as np
import resampy
import soundfile as sf
from queue import Queue
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from assistant.config import (
    SPEECH_PIPELINE_BUFFER_SIZE_MILIS,
    SPEECH_PIPELINE_SAMPLERATE,
)
from assistant.core.component import Component
from assistant.utils.audio import VadFilter, chop_audio
from assistant.utils.audio.reshape import FixedLengthAudioChunker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, EVENT_TYPE_CREATED

from assistant.utils.utils import observe


from . import events
from assistant.components import watchdog
from assistant.components.mumble.mumble import SpeechSegment

class WatchDirectory(BaseModel):
    path: str
    recursive: bool = Field(default=False)
    extensions: List[str] = Field(default_factory=list)


class FileEvents(Enum):
    FILE_CREATED = auto()


class SimpleHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str, bytes | str], None]) -> None:
        super().__init__()
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory:
            self.callback(event.event_type, event.src_path)


class Watchdog(Component):
    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return [events.WATCHDOG_FILE_NEW, events.WATCHDOG_FILE_AUDIO, events.WATCHDOG_AUDIO_SPEECH_DETECTED]

    def initialize(self) -> None:
        super().initialize()
        self.logger.setLevel(self.get_config("log_level", "DEBUG"))
        watch_list = self.get_config("watch", [])
        assert isinstance(watch_list, list)

        observer = Observer()
        for item in watch_list:
            item = WatchDirectory.model_validate(item)
            if os.path.exists(item.path):
                event_handler = SimpleHandler(self.on_file)
                observer.schedule(event_handler, item.path, recursive=item.recursive)
            else:
                self.logger.warning(f"Directory '{item.path}' not found.")

        observer.start()
        self.file_events = Queue()
        self.file_events_observer = observe(
            self.file_events, lambda item: self.categorize_files(*item)
        )
        self.vad_filter = VadFilter(self.on_speech)

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' disconnection from server.")

    def on_file(self, event: str, file: str | bytes):
        self.file_events.put_nowait((event, file))

    @staticmethod
    def is_audio(file: str):
        return bool(
            list(filter(lambda e: file.endswith(e), ["flac", "wav", "ogg", "mp3"]))
        )

    def categorize_files(self, event: str, file: str | bytes):
        if EVENT_TYPE_CREATED == event and self.is_audio(str(file)):
            self.process_audio(file)

    def process_audio(self, file: str | bytes):
        self.logger.info(f"Starting processing of '{file}' audio file")
        sound, samplerate = sf.read(file)

        resampled_float = resampy.resample(
            sound, samplerate, SPEECH_PIPELINE_SAMPLERATE
        )
        resampled = (resampled_float * np.iinfo(np.int16).max).astype(np.int16)

        for segment in chop_audio(
            resampled, SPEECH_PIPELINE_SAMPLERATE, SPEECH_PIPELINE_BUFFER_SIZE_MILIS
        ):
            self.vad_filter(segment)  # TODO: Logging

        self.logger.info(f"Processing of '{file}' audio file done")

    def on_speech(self, speech: bytes):
        self.logger.info(f"{type(speech)}")

        data = np.frombuffer(speech, dtype=np.int16)

        segment = SpeechSegment(source="watchdog", data=data)
        self.proxy(events.WATCHDOG_AUDIO_SPEECH_DETECTED)(segment)
