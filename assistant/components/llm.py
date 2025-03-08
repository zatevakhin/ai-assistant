from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import threading
import logging
from queue import Queue
from typing import List, Any
from .event_bus import EventBus, EventType
from .transcriber import TranscribedSegment

from pydantic import BaseModel, Field

from assistant.config import (
    OLLAMA_LLM_STOP_TOKENS,
    OLLAMA_LLM_TEMPERATURE,
    OLLAMA_LLM,
    OLLAMA_BASE_URL,
    ASSISTANT_BREAK_ON_TOKENS,
    INITIAL_SYSTEM_PROMPT,
    INTERRUPT_PROMPT,
)
from .util import queue_as_observable, controlled_area, ensure_model_exists
import reactivex as rx
from reactivex import operators as ops
from functools import partial


logger = logging.getLogger(__name__)


class StreamToken(BaseModel):
    token: str = Field(repr=True)
    done: bool = Field(repr=True)

class QueryResponse(BaseModel):
    tokens: List[StreamToken] = Field(repr=True, default_factory=list)
    interrupted: bool = Field(repr=True)


class LlmInferenceProcess:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus

        self.query_subscription = self.event_bus.subscribe(EventType.TRANSCRIPTION_DONE, self.on_query)
        self.interruption_subscription = self.event_bus.subscribe(EventType.LLM_STREAM_INTERRUPT, lambda _: self.on_interruption(None))

        self.queries: Queue[str] = Queue()
        self.observable_queries = queue_as_observable(self.queries)
        self.observable_queries.subscribe(self.__query_handler)
        self.publish_sentence = partial(self.event_bus.publish, EventType.LLM_NEW_SENTENCE)

        self.llm = self.create_llm(OLLAMA_LLM)
        self.running = False
        self.interrupt_inference = threading.Event()

        self.history = [
            SystemMessage(content=INITIAL_SYSTEM_PROMPT),
        ]

        output_parser = StrOutputParser()
        self.chain = self.llm.bind(stop=["user:", *OLLAMA_LLM_STOP_TOKENS]) | output_parser

        logger.info("LLM Inference ... IDLE")

    @staticmethod
    def create_llm(model: str):
        ensure_model_exists(OLLAMA_BASE_URL, model)

        logger.info(f"Creating Ollama using '{model}' model.")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=OLLAMA_LLM_TEMPERATURE,
        )

    def __query_handler(self, query: str):
        with controlled_area(partial(self.event_bus.publish, EventType.LLM_INFERENCE_STATUS), "running", "done", True, __name__):
            self.history.append(HumanMessage(content=query))
            query_response = QueryResponse(tokens=[], interrupted=False)

            rx.from_iterable(self.chain.stream(self.history)).pipe(
                ops.map(lambda t: StreamToken(token=t, done=bool(t == ""))),
                ops.do_action(query_response.tokens.append),
                ops.publish(lambda shared: shared.pipe(
                    ops.buffer_when(lambda: shared.pipe(
                        ops.filter(lambda t: not t.done and t.token[-1] in ASSISTANT_BREAK_ON_TOKENS),
                        ops.take(1)
                    ))
                )),
                ops.take_while(lambda _: not self.interrupt_inference.is_set()),
                ops.do_action(self.__on_new_buffer),
                ops.take_while(lambda t: len(t) and not t[-1].done),
                ops.finally_action(lambda: self.__on_done(query_response)),
            ).run()

    def __on_new_buffer(self, buffer: List[StreamToken]):
        sentence = "".join(map(lambda t: t.token, buffer))
        if len(sentence):
            self.publish_sentence(sentence)

    def __on_done(self, response: QueryResponse):
        response.interrupted = self.interrupt_inference.is_set()

        full_response = "".join(map(lambda t: t.token, response.tokens))
        self.history.append(AIMessage(content=full_response))

        if response.interrupted:
            self.history.append(SystemMessage(content=INTERRUPT_PROMPT))
        self.event_bus.publish(EventType.LLM_STREAM_DONE, response)

    def on_query(self, t: TranscribedSegment):
        logger.info(f"on_query({t})")
        if t.language not in ["en"]:
            logger.warning(f"Language '{t.language}' will not be handled.")
            return

        min_probability = 0.6
        if t.probability < min_probability:
            logger.warning(f"Probability that this is '{t.language}' language is below '{min_probability}'.")
            return

        self.queries.put(t.text)

    def on_interruption(self, sample: Any):
        logger.warning(f"Interrupting: {sample}")
        self.interrupt_inference.set()

    def run(self):
        self.running = True
        logger.info("LLM Inference ... OK")

    def stop(self):
        logger.info("LLM Inference ... STOPPING")
        self.running = False
        self.query_subscription.dispose()
        self.interruption_subscription.dispose()
        logger.info("LLM Inference ... DEAD")
