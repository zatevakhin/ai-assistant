import logging
from assistant.components.llm import QueryResponse
from assistant.components.synthesis import Sentence
from queue import Queue
from .util import ensure_model_exists
from .event_bus import EventBus, EventType
from langchain_ollama import ChatOllama
from .transcriber import TranscribedSegment
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


from assistant.config import (
    ASSISTANT_NAME,
    OLLAMA_LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_KEEP_ALIVE
)

logger = logging.getLogger(__name__)

FILTER_PROMPT_TEMPLATE = f"""
You are {ASSISTANT_NAME}, deciding how to handle a new user input while in an ongoing conversation. Prioritize the user's needs while maintaining focus on the current task. Be concise in your answer.

Previous input: {{previous_query}}
Current input: {{current_query}}

Select ONE action:

1. DISCARD (Default): Continue with previous input, ignoring current input.
   Use when current input is not urgent or interrupting.

2. INTERRUPT: Stop current process and address new input immediately.
   Use when user says WAIT, STOP, or explicitly requests interruption.

3. APPEND: Combine both inputs and address together.
   Use when current input directly relates to and enhances previous input.

4. PARALLEL: Handle both inputs simultaneously.
   Use ONLY for urgent inputs that cannot be discarded while previous remains important.
"""

class ActionDecision(BaseModel):
    action: str = Field(description="The chosen action: DISCARD, INTERRUPT, APPEND or PARALLEL")
    reason: str = Field(description="A brief explanation for why this action was chosen")


class InterruptOr:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.event_bus.subscribe(EventType.TRANSCRIPTION_DONE, self.on_query)
        self.event_bus.subscribe(EventType.LLM_INFERENCE_STATUS, self.on_inference_status)
        self.event_bus.subscribe(EventType.LLM_STREAM_DONE, self.on_llm_stream_done)
        self.event_bus.subscribe(EventType.MUMBLE_PLAYING_STATUS, self.on_mumble_playing_status)
        self.event_bus.subscribe(EventType.MUMBLE_NOW_PLAYING, self.on_mumble_now_playing)
        self.event_bus.subscribe(EventType.TRANSCRIPTION_STATUS, self.on_transcription_status)
        self.event_bus.subscribe(EventType.SPEECH_SYNTH_STATUS, self.on_speech_synthesis_status)

        self.llm = self.create_llm("llama3.2:3b")

        self.running = False
        self.queries: Queue[TranscribedSegment] = Queue()
        logger.info(f"{__name__} ... IDLE")

    @staticmethod
    def create_llm(model: str):
        ensure_model_exists(OLLAMA_BASE_URL, model)

        logger.info(f"Creating Ollama using '{model}' model.")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=OLLAMA_LLM_TEMPERATURE,
            keep_alive=OLLAMA_MODEL_KEEP_ALIVE,
        )

    def on_query(self, query: TranscribedSegment):
        logger.info(f"on_query = {query}")

        if query.probability <= 0.6:
            logger.info(f"query = '{query}', is ignored")
            return

        if not self.queries.empty():
            previous_query = self.queries.get()

            prompt = PromptTemplate(
                template=FILTER_PROMPT_TEMPLATE,
                input_variables=["previous_query", "current_query"],
            )

            filter_prompt = prompt.format_prompt(previous_query=previous_query.text, current_query=query.text)
            result: ActionDecision = self.llm.with_structured_output(ActionDecision).invoke(filter_prompt)

            logger.info(f"-> {result}")

            if result.action in ["DISCARD"]:
               self.event_bus.publish(EventType.LLM_SENTENCE_DISCARD, query.uuid)
               self.queries.task_done()
            elif result.action in ["INTERRUPT"]:
               self.event_bus.publish(EventType.LLM_STREAM_INTERRUPT, query.uuid)
               self.queries.task_done()
            else:
               logger.warning(f"action = '{result.action}', is not handled")

        elif self.queries.empty():
            self.queries.put(query)

    def on_inference_status(self, query: str):
        logger.info(f"on_inference_status('{query}')")

    def on_llm_stream_done(self, query: QueryResponse):
        logger.info(f"on_llm_stream_done({query}, {len(query.tokens)})")

    def on_mumble_playing_status(self, query: str):
        logger.info(f"on_mumble_playing_status('{query}')")

    def on_mumble_now_playing(self, sentence: Sentence):
        logger.info(f"on_mumble_now_playing({sentence})")

    def on_transcription_status(self, query: str):
        logger.info(f"on_transcription_status('{query}')")

    def on_speech_synthesis_status(self, query: str):
        logger.info(f"on_speech_synthesis_status('{query}')")

    def run(self):
        self.running = True
        logger.info(f"{__name__} ... OK")

    def stop(self):
        logger.info(f"{__name__} ... STOPPING")
        self.running = False
        logger.info(f"{__name__} ... DEAD")
