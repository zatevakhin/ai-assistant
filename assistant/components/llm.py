from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

import threading
import logging
import zenoh
from queue import Queue, Empty
from zenoh import Sample
from typing import Callable, List
import json
from .event_bus import EventBus

from pydantic import BaseModel

from assistant.config import (
    ASSISTANT_NAME,
    OLLAMA_LLM_STOP_TOKENS,
    OLLAMA_LLM_TEMPERATURE,
    OLLAMA_LLM,
    OLLAMA_BASE_URL,
    TOPIC_TRANSCRIPTION_DONE,
    TOPIC_LLM_TOKEN_NEW,
    TOPIC_LLM_STREAM_DONE,
    TOPIC_LLM_STREAM_INTERRUPT,
    TOPIC_SPEECH_SYNTHESIS_INTERRUPT,
    TOPIC_MUMBLE_INTERRUPT_AUDIO,
    TOPIC_LLM_ON_SENTENCE,
    ZENOH_CONFIG,
)

logger = logging.getLogger(__name__)


class StreamToken(BaseModel):
    token: str
    done: bool

class QueryResponse(BaseModel):
    tokens: List[StreamToken]
    interrupted: bool


class TokenBuffer:
    def __init__(self, callback: Callable[[StreamToken], None]):
        self.buffer: List[StreamToken] = []
        self.callback = callback
        self.pause_tokens = {'.', ',', '!', '?', ";", "\n"}

    def add(self, t: StreamToken):
        self.buffer.append(t)
        if t.token in self.pause_tokens or t.done:
            self.flush()

    def flush(self):
        sentence = ''.join(map(lambda t: t.token, self.buffer)).strip()
        if sentence:
            self.callback(sentence)
        self.buffer.clear()

    def reset(self):
        self.buffer.clear()


class LlmInferenceProcess:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus

        self.query_subscription = self.event_bus.subscribe(TOPIC_TRANSCRIPTION_DONE, self.on_query)

        self.zenoh_session = zenoh.open(ZENOH_CONFIG)
        # self.sub_on_query = self.zenoh_session.declare_subscriber(TOPIC_TRANSCRIPTION_DONE, self.on_query)
        self.sub_llm_stream_interrupt = self.zenoh_session.declare_subscriber(TOPIC_LLM_STREAM_INTERRUPT, self.on_interruption)
        self.pub_interrupt = self.zenoh_session.declare_publisher(TOPIC_LLM_STREAM_INTERRUPT)
        self.pub_interrupt_speech_synth = self.zenoh_session.declare_publisher(TOPIC_SPEECH_SYNTHESIS_INTERRUPT)
        self.pub_interrupt_speech_playback = self.zenoh_session.declare_publisher(TOPIC_MUMBLE_INTERRUPT_AUDIO)
        self.pub_llm_stream_done = self.zenoh_session.declare_publisher(TOPIC_LLM_STREAM_DONE)
        self.sub_on_new_token = self.zenoh_session.declare_subscriber(TOPIC_LLM_TOKEN_NEW, self.on_token)
        self.sub_on_query_done = self.zenoh_session.declare_subscriber(TOPIC_LLM_STREAM_DONE, self.on_query_done)
        self.pub_on_sentence = self.zenoh_session.declare_publisher(TOPIC_LLM_ON_SENTENCE)
        self.pub_llm_token_new = self.zenoh_session.declare_publisher(TOPIC_LLM_TOKEN_NEW)

        self.token_buff = TokenBuffer(self.on_sentence)

        self.llm = self.create_llm(OLLAMA_LLM)
        self.queries: Queue[str] = Queue()
        self.running = False
        self.thread = threading.Thread(target=self._chat_with_llm)
        self.interrupt_inference = threading.Event()
        self.is_chatting = threading.Event()


        self.messages = [
            SystemMessage(content=f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise."),
        ]

        output_parser = StrOutputParser()
        self.chain = self.llm.bind(stop=["user:", *OLLAMA_LLM_STOP_TOKENS]) | output_parser

        logger.info("LLM Inference ... IDLE")

    @staticmethod
    def create_llm(model: str):
        logger.info(f"Creating Ollama using '{model}' model.")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=OLLAMA_LLM_TEMPERATURE,
        )

    def _chat_with_llm(self):
        while self.running:
            try:
                text = self.queries.get(timeout=1)
                self.is_chatting.set()
                logger.info(f">>> \"{text}\"")
            except Empty:
                continue

            self.messages.append(HumanMessage(content=text))

            tokens: List[StreamToken] = []
            for token in self.chain.stream(self.messages):

                if self.interrupt_inference.is_set():
                    stream_done = QueryResponse(tokens=tokens, interrupted=self.interrupt_inference.is_set())
                    self.pub_llm_stream_done.put(stream_done.model_dump())
                    self.interrupt_inference.clear()
                    self.token_buff.reset()
                    break

                s_token = StreamToken(token=token, done=bool(token == ""))
                self.pub_llm_token_new.put(s_token.model_dump())
                tokens.append(s_token)
            else:
                stream_done = QueryResponse(tokens=tokens, interrupted=self.interrupt_inference.is_set())
                self.pub_llm_stream_done.put(stream_done.model_dump())

            # Always clear state, after chat finished or interrupted.
            self.is_chatting.clear()


    def on_query(self, transcription: dict):
        logger.info(f"on_query({transcription})")
        language = transcription["language"]
        text = transcription["text"]
        probability = transcription["probability"]

        if language not in ["en"]:
            logger.warning(f"Language '{language}' will not be handled.")
            return

        min_probability = 0.6
        if probability < min_probability:
            logger.warning(f"Probability that this is '{language}' language is below '{min_probability}'.")
            return

        if self.is_chatting.is_set():
            logger.warning("Interrupting because LLM inference is running.")
            self.messages.append(HumanMessage(content=text))
            self.pub_interrupt.put({})
            return

        self.queries.put(text)

    def on_interruption(self, sample: Sample):
        logger.warning(f"Interrupting: {sample.payload}")
        self.interrupt_inference.set()
        self.pub_interrupt_speech_synth.put({"by": __name__})
        self.pub_interrupt_speech_playback.put({"by": __name__})

    def on_token(self, sample: Sample):
        token = StreamToken.model_validate_json(sample.payload)
        logger.debug(f"> {token}")
        self.token_buff.add(token)

    def on_sentence(self, sentence: str):
        logger.info(f"--> {sentence}")
        self.pub_on_sentence.put(sentence.encode())

    def on_query_done(self, sample: Sample):
        response = QueryResponse.model_validate_json(sample.payload)
        logger.debug(f">>> {response}")

        ai_response = "".join(map(lambda t: t.token, response.tokens)).strip()
        self.messages.append(AIMessage(content=ai_response))

        if response.interrupted:
            self.messages.append(SystemMessage(content=f"Note, {ASSISTANT_NAME}, you were interrupted by a user with previous message."))

    def run(self):
        self.running = True
        self.thread.start()
        logger.info("LLM Inference ... OK")

    def stop(self):
        logger.info("LLM Inference ... STOPPING")
        self.running = False
        self.thread.join()
        self.query_subscription.dispose()
        logger.info("LLM Inference ... DEAD")
