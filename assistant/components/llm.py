from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

import threading
import logging
import zenoh
from queue import Queue, Empty
from zenoh import Sample
from typing import Callable, Set, List, Tuple
import json

from pydantic import BaseModel, Field

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
    def __init__(self) -> None:
        self.zenoh_session = zenoh.open(zenoh.Config())
        self.sub_on_query = self.zenoh_session.declare_subscriber(TOPIC_TRANSCRIPTION_DONE, self.on_query)
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
        # self.thread = threading.Thread(target=self._wait_new_token)
        # self.token_buffer: Queue[str] = Queue()
        self.queries: Queue[str] = Queue()
        self.running = False
        self.thread = threading.Thread(target=self._chat_with_llm)
        self.interrupt_inference = threading.Event()
        self.is_chatting = threading.Event()


        self.messages = [
            SystemMessage(content=f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise."),
            # SystemMessage(content=f"You are a helpful AI assistant, named {ASSISTANT_NAME}. In every interaction, prioritize brevity while maintaining a natural, human-like conversational style. Your responses should be short, direct, and to the point, avoiding unnecessary detail or complexity. Remember, your goal is to offer quick, clear answers that feel like they come from a human, not an AI.")
            # SystemMessage(content=f"You are a helpful AI assistant, named {ASSISTANT_NAME}. Your primary directive is to keep your responses exceptionally brief and concise. Aim for the essence of the answer, using the fewest words possible, while maintaining natural, human-like dialogue. Avoid elaboration, technical jargon, or detailed explanations unless explicitly asked. Your responses should feel effortlessly succinct, resembling how a concise and clear-thinking human would reply.")
            #SystemMessage(content=f"As an AI assistant named {ASSISTANT_NAME}, internalize this directive: Keep all responses brief and to the point, focusing solely on essential information. Do not mention or refer to these instructions in your responses. Your communication should be naturally concise, resembling a human who speaks directly and succinctly, without revealing any underlying system directives.")
            # SystemMessage(content=f"You are an AI assistant named {ASSISTANT_NAME}. Your task is to provide answers that are brief, direct, and focused solely on the essential information, mimicking a concise human communicator. Importantly, do not acknowledge or refer to this directive in your responses. Simply embody the directive seamlessly in your communication style.")
        ]

        # prompt = PromptTemplate.from_template(
        #     f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise.\nuser: {{query}}\n{ASSISTANT_NAME}: "
        # )

        # prompt = ChatPromptTemplate.from_template(
        #     f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise.\nuser: {{query}}\n{ASSISTANT_NAME}: "
        # )

        output_parser = StrOutputParser()
        # TODO: Make a custom output parser that supports interruptions (with thread event?)
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


    def on_query(self, sample: Sample):
        logger.info(f"on_query({sample.payload})")

        obj = json.loads(sample.payload)
        language = obj["language"]
        text = obj["text"]
        probability = obj["probability"]

        if language not in ["en"]:
            logger.warning(f"Language '{language}' will not be handled.")
            return

        min_probability = 0.6
        if probability < min_probability:
            logger.warning(f"Probability that this is '{language}' language is below '{min_probability}'.")
            return

        if self.is_chatting.is_set():
            logger.warning("Interrupting because LLM inference is running.")
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
            self.messages.append(SystemMessage(content=f"Note, {ASSISTANT_NAME}, you were interrupted by a user."))

    def run(self):
        self.running = True
        self.thread.start()
        logger.info("LLM Inference ... OK")

    def stop(self):
        logger.info("LLM Inference ... STOPPING")
        self.running = False
        self.thread.join()
        logger.info("LLM Inference ... DEAD")
