import time
import logging
from rich.logging import RichHandler
from assistant.components import (
    VadProcess,
    MumbleProcess,
    SpeechTranscriberProcess,
    LlmInferenceProcess,
    SpeechSynthesisProcess,
)


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self) -> None:
        self.mumble = MumbleProcess()
        self.vad = VadProcess()
        self.transcriber = SpeechTranscriberProcess()
        self.llm_inference = LlmInferenceProcess()
        self.speech_synthesis = SpeechSynthesisProcess()

    def run(self) -> None:
        self.mumble.run()
        self.vad.run()
        self.transcriber.run()
        self.llm_inference.run()
        self.speech_synthesis.run()

        time.sleep(600)

        self.mumble.stop()
        self.transcriber.stop()
        self.llm_inference.stop()
        self.speech_synthesis.stop()
        self.vad.stop()



def main() -> None:
    logger.info("Initializing Voice Assistant ...")
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
