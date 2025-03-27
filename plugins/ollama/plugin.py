import threading
from typing import List

from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    StrOutputParser,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from assistant.config import ASSISTANT_NAME

from assistant.core import Plugin, service
from assistant.utils import ensure_model_exists, event_context


INITIAL_SYSTEM_PROMPT = f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise."


class StreamToken(BaseModel):
    token: str = Field(repr=True)
    done: bool = Field(repr=True)


class QueryResponse(BaseModel):
    tokens: List[StreamToken] = Field(default_factory=list)
    interrupted: bool

    def add(self, t):
        self.tokens.append(StreamToken(token=t, done=bool(t == "")))


class Ollama(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return []

    def initialize(self) -> None:
        super().initialize()
        model = self.get_config("model", "llama3.2:3b")

        self.llm = self.create_llm(model)
        self.chain = self.llm | StrOutputParser()

        self.history: List[BaseMessage] = [
            SystemMessage(content=INITIAL_SYSTEM_PROMPT),
        ]

        self.is_inferencing = threading.Event()
        self.is_interrupt = threading.Event()
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def create_llm(self, model: str):
        url = self.get_config("url", "localhost:11434")
        temperature = self.get_config("temperature", 0.0)
        ensure_model_exists(url, model)

        self.logger.info(f"Creating Ollama using '{model}' model.")
        return ChatOllama(
            base_url=url,
            model=model,
            temperature=temperature,
        )

    @service
    async def interrupt(self):
        self.logger.info("interrupt")
        if self.is_inferencing.is_set():
            self.is_interrupt.set()

    @service
    def ask(self, query: str):
        self.history.append(HumanMessage(content=query))

        with event_context(self.is_inferencing):
            query_response = QueryResponse(tokens=[], interrupted=False)

            for token in self.chain.stream(self.history):
                if self.is_interrupt.is_set():
                    self.is_interrupt.clear()
                    query_response.interrupted = True
                    break
                query_response.add(token)
                yield token

            response = "".join(map(lambda t: t.token, query_response.tokens))
            self.history.append(AIMessage(content=response))
