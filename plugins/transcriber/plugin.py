from typing import List
import requests
import io
from assistant.core import Plugin, service
from voice_pulse import SpeechSegment
from assistant.config import SPEECH_PIPELINE_SAMPLERATE, WHISPERX_API_URL, WHISPERX_MODEL_NAME
import soundfile as sf
import numpy as np

from .types import Transcript

class TranscriberService(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return []

    def initialize(self) -> None:
        super().initialize()

        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    @service
    async def transcribe_speech(self, segment: SpeechSegment):
        self.logger.info(f"> transcribe_speech({type(segment)})")

        print(await self.transcribe(segment))

    async def transcribe(self, segment: SpeechSegment):
        """Send speech segment to URL endpoint for transcription"""

        try:
            audio = io.BytesIO()
            sf.write(audio, segment.speech.astype(np.float32), SPEECH_PIPELINE_SAMPLERATE, format="FLAC")
            audio.seek(0)

            response = requests.post(
                f"{WHISPERX_API_URL}/transcribe",
                files={"file": ("audio.flac", audio, f"audio/flac")},
                data={"whisper_model": WHISPERX_MODEL_NAME, "diarize": True, "align_words": True}
            )

            if not response.status_code == 200:
                raise requests.exceptions.HTTPError(f"Transcription failed with status code: {response.status_code}")

            return Transcript.model_validate(response.json())

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to process transcription request: {str(e)}")
            raise Exception("Transcription failed due to network or connection issues")


