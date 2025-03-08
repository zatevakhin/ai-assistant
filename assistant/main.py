import time
import logging
from rich.logging import RichHandler
from assistant.components import (
    VadProcess,
    MumbleProcess,
    SpeechTranscriberProcess,
    LlmInferenceProcess,
    SpeechSynthesisProcess,
    EventBus,
    InterruptOr
)

from assistant.components.graph import SystemIII


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self) -> None:
        self.event_bus = EventBus()

        self.mumble = MumbleProcess(self.event_bus)
        self.vad = VadProcess(self.event_bus)
        # TODO: Combine speech input if there is not a big delta.
        self.transcriber = SpeechTranscriberProcess(self.event_bus)
        # TODO: Consider to add interrupt manager process, that will evaluate if interruption needed or not.
        self.interruptor = InterruptOr(self.event_bus)
        #self.llm_inference = LlmInferenceProcess(self.event_bus)
        self.speech_synthesis = SpeechSynthesisProcess(self.event_bus)
        self.system_iii = SystemIII(self.event_bus)

    def run(self) -> None:
        self.mumble.run()
        self.vad.run()
        self.transcriber.run()
        self.interruptor.run()
        #self.llm_inference.run()
        self.speech_synthesis.run()
        self.system_iii.run()

        time.sleep(600)

        self.mumble.stop()
        self.transcriber.stop()
        self.interruptor.stop()
        #self.llm_inference.stop()
        self.speech_synthesis.stop()
        self.vad.stop()
        self.system_iii.stop()



def main() -> None:
    logger.info("Initializing Voice Assistant ...")
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
