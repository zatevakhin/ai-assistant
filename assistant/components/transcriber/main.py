from datetime import datetime
import io
from queue import Queue
from typing import List, Optional

import requests
import soundfile as sf

from assistant.config import (
    SPEECH_PIPELINE_SAMPLERATE,
)
from assistant.core import service
from assistant.core.component import Component
from assistant.components.mumble.mumble import SpeechSegment
from assistant.core.config_manager import ConfigManager
from assistant.utils.utils import observe
from .types import Transcript
from .events import (
    TRANSCRIPTION_SEGMENT_STARTED,
    TRANSCRIPTION_QUEUE_ADDED,
    TRANSCRIPTION_SEGMENT_DONE,
)


class TranscriberService(Component):
    def __init__(
        self, name: Optional[str] = None, config: Optional[ConfigManager] = None
    ):
        super().__init__(name or "transcriber", config)

    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return [
            TRANSCRIPTION_SEGMENT_DONE,
            TRANSCRIPTION_QUEUE_ADDED,
            TRANSCRIPTION_SEGMENT_STARTED,
        ]

    def initialize(self) -> None:
        super().initialize()
        self.speech_segments = Queue()
        self.speech_segments_observer = observe(self.speech_segments, self.transcribe_segment, threaded=True, max_workers=4)

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.speech_segments_observer.dispose()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def on_speech(self, segment: SpeechSegment):
        self.speech_segments.put_nowait(segment)

    def transcribe_segment(self, segment: SpeechSegment):
        self.logger.info(f"-> {datetime.now() - segment.timestamp}")
        self.proxy(TRANSCRIPTION_SEGMENT_STARTED)(segment)
        whisperx = self.get_config("whisperx", {})

        try:
            audio = io.BytesIO()
            sf.write(
                audio,
                segment.data,
                SPEECH_PIPELINE_SAMPLERATE,
                format="FLAC",
            )
            audio.seek(0)

            url = whisperx.get("url", "http://localhost:8000")

            response = requests.post(
                f"{url}/transcribe",
                files={"file": ("audio.flac", audio, "audio/flac")},
                data={
                    "whisper_model": whisperx.get("model", "small"),
                    "diarize": whisperx.get("diarize", False),
                    "align_words": whisperx.get("align", False),
                },
            )

            if not response.status_code == 200:
                raise requests.exceptions.HTTPError(
                    f"Transcription failed with status code: {response.status_code}"
                )

            transcript = Transcript.model_validate(response.json())
            self.proxy(TRANSCRIPTION_SEGMENT_DONE)(segment, transcript)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to process transcription request: {str(e)}")
            raise Exception("Transcription failed due to network or connection issues")
