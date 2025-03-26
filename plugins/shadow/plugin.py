from enum import Enum
import threading
from langchain.schema import HumanMessage, StrOutputParser, SystemMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from assistant.core import Plugin, service
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from assistant.utils import ensure_model_exists, event_context
from assistant.config import (
    OLLAMA_LLM_STOP_TOKENS,
    OLLAMA_LLM_TEMPERATURE,
    OLLAMA_LLM,
    OLLAMA_BASE_URL,
    ASSISTANT_BREAK_ON_TOKENS,
    INITIAL_SYSTEM_PROMPT,
    INTERRUPT_PROMPT,
)

from plugins.ollama.plugin import StreamToken

DECISION_SYSTEM_PROMPT = """
You are an AI assistant that processes transcription chunks. Your job is to:

1. Analyze each transcription chunk to determine its importance and relevance
2. Decide what action to take with the chunk based on its content
3. Provide a clear reason for your decision

For each chunk, decide between these actions:
- ADD_TO_CONTEXT: Add to the current conversation context because it contains relevant information for the ongoing dialogue
- STORE_IN_MEMORY: Store as a memory/fact for later use because it contains important information but isn't immediately relevant
- DISCARD: Discard because it's not important (e.g., filler words, background noise transcription)
"""

class ActionDecision(BaseModel):
    action: str = Field(description="The chosen action ADD_TO_CONTEXT, STORE_IN_MEMORY or DISCARD")
    reason: str = Field(description="A brief explanation for why this action was chosen")

class QueryResponse(BaseModel):
    tokens: List[StreamToken] = Field(default_factory=list)
    interrupted: bool

    def add(self, t):
        self.tokens.append(StreamToken(token=t, done=bool(t == "")))

class TranscriptionAction(str, Enum):
    ADD_TO_CONTEXT = "ADD_TO_CONTEXT"
    STORE_IN_MEMORY = "STORE_IN_MEMORY"
    DISCARD = "DISCARD"


class Shadow(Plugin):
    @property
    def version(self) -> str:
        return "0.0.1"

    def get_event_definitions(self) -> List[str]:
        return []

    def initialize(self) -> None:
        super().initialize()
        # TODO: Make it configurable
        self.llm = self.create_llm("llama3.2:3b")

        self.history: List[BaseMessage] = [
            SystemMessage(content=INITIAL_SYSTEM_PROMPT),
        ]

        # Conversation context that we're building
        self.context: List[str] = []

        self.is_processing = threading.Event()
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

    def shutdown(self) -> None:
        super().shutdown()
        self.logger.info(f"Plugin '{self.name}' shutdown done.")

    def create_llm(self, model: str):
        ensure_model_exists(OLLAMA_BASE_URL, model)

        self.logger.info(f"Creating Ollama using '{model}' model.")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=OLLAMA_LLM_TEMPERATURE,
        )

    @service
    async def receive(self, transcription_chunk: str):
        """
        Receive a transcription chunk from external service and process it

        """
        self.logger.info(f"Received transcription chunk: {transcription_chunk[:50]}...")

        with event_context(self.is_processing):
            if self.context:
                combined_context = " ".join(self.context + [transcription_chunk])
            else:
                combined_context = transcription_chunk

            messages = [
                SystemMessage(content=DECISION_SYSTEM_PROMPT),
                HumanMessage(content=f"Transcription chunk: \"{combined_context}\"")
            ]


            decision: ActionDecision = self.llm.with_structured_output(ActionDecision).invoke(messages)

            if decision.action == TranscriptionAction.ADD_TO_CONTEXT:
                self.context.append(transcription_chunk)

            elif decision.action == TranscriptionAction.STORE_IN_MEMORY:
                self.context = []
            elif decision.action == TranscriptionAction.DISCARD:
                pass

            return decision



