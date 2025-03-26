from typing import List

from assistant.core import Plugin, service
from voice_pulse import SpeechSegment


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



