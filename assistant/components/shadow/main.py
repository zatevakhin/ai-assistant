from datetime import datetime
from queue import Queue
import threading
from enum import Enum
from typing import List, Optional

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from assistant.components.mumble.mumble import SpeechSegment
from assistant.components.transcriber.types import Transcript
from assistant.core.component import Component
from assistant.utils import ensure_model_exists, event_context
from assistant.utils.utils import observe
from plugins.ollama.plugin import StreamToken

DECISION_SYSTEM_PROMPT = """
You are an AI assistant that processes transcription chunks. Your job is to:

1. Analyze each transcription chunk to determine its importance and relevance
2. Decide what action to take with the chunk based on its content
3. Provide a clear reason for your decision
4. Be atentive to details, some parts of transcription might be transcribed incorrectly
5. If you have any mismatch in your context and it was clarified later you should correct yourself

For each chunk, decide between these actions:
- ADD_TO_CONTEXT: Add to the current conversation context because it contains relevant information for the ongoing dialogue
- STORE_IN_MEMORY: Store as a memory/fact for later use because it contains important information but isn't immediately relevant
- DISCARD: Discard because it's not important (e.g., filler words, background noise transcription)
"""


class ActionDecision(BaseModel):
    action: str = Field(
        description="The chosen action ADD_TO_CONTEXT, STORE_IN_MEMORY or DISCARD"
    )
    reason: str = Field(
        description="A concise explanation for why this action was chosen"
    )


CONDENSED_MEMORY_PROMPT = """
You are an AI assistant that processes transcription context to create concise, meaningful memory summaries. Your job is to:

1. Synthesize the accumulated context into a coherent summary
2. Extract key entities, topics, and relationships mentioned in the conversation
3. Identify any explicit or implicit user requests, questions, or needs
4. Preserve important factual details, especially names, dates, numbers, and specific references
5. Distinguish between definitive statements and uncertain or speculative information
6. Maintain conversational tone and intent where relevant
7. Resolve any transcription errors or ambiguities when the meaning becomes clear
8. Capture emotional content or sentiment when significant to the interaction

Remember that these summaries will be stored as persistent memory that can be referenced later, so they should:
- Be self-contained and comprehensible without additional context
- Include sufficient detail to be useful when retrieved
- Prioritize accuracy over brevity for critical information
- Use clear, precise language that avoids ambiguity

Create a summary that captures the essence of the current context while preserving all actionable and memorable content.
"""


class MemorySummary(BaseModel):
    summary: str = Field(
        description="A concise yet comprehensive summary of the accumulated context"
    )
    entities: List[str] = Field(
        description="Key entities (people, places, objects, concepts) mentioned in the context",
        default_factory=list
    )
    topics: List[str] = Field(
        description="Main topics or themes discussed in the context",
        default_factory=list
    )

class QueryResponse(BaseModel):
    tokens: List[StreamToken] = Field(default_factory=list)
    interrupted: bool

    def add(self, t):
        self.tokens.append(StreamToken(token=t, done=bool(t == "")))


class TranscriptionAction(str, Enum):
    ADD_TO_CONTEXT = "ADD_TO_CONTEXT"
    STORE_IN_MEMORY = "STORE_IN_MEMORY"
    DISCARD = "DISCARD"


class Shadow(Component):
    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def events(self) -> List[str]:
        return []

    def initialize(self) -> None:
        super().initialize()
        model = self.get_config("model", "llama3.2:3b")
        self.llm = self.create_llm(model)

        # Conversation context that we're building
        self.context: List[str] = []

        self.is_processing = threading.Event()
        self.logger.info(f"Plugin '{self.name}' initialized and ready")

        self.transcripts = Queue()
        observe(self.transcripts, lambda args: self.process_transcript(*args))



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

    def on_transcript(self, segment: SpeechSegment, transcript: Transcript):
        self.transcripts.put_nowait((segment, transcript))


    def process_transcript(self, segment: SpeechSegment, t: Transcript):
        self.logger.info(f"-> {datetime.now() - segment.timestamp}")

        self.logger.info(f"({t.language}: {t.duration}) -> {t.transcript}...")

        with event_context(self.is_processing):
            if self.context:
                combined_context = " ".join(self.context + [t.transcript])
            else:
                combined_context = t.transcript

            messages = [
                SystemMessage(content=DECISION_SYSTEM_PROMPT),
                HumanMessage(content=f'Transcription chunk: "{combined_context}"'),
            ]

            decision: Optional[ActionDecision] = self.llm.with_structured_output(
                ActionDecision
            ).invoke(messages)

            self.last_decision = decision
            if decision is None:
                return

            self.logger.info(f"[{decision.action}] {decision.reason}")

            if decision.action == TranscriptionAction.ADD_TO_CONTEXT:
                self.context.append(t.transcript)

            elif decision.action == TranscriptionAction.STORE_IN_MEMORY:
                self.logger.info("[MEM_DUMP]")

                context = "\r\n -".join(self.context)

                messages = [
                    SystemMessage(content=CONDENSED_MEMORY_PROMPT),
                    HumanMessage(content=f"Context: {context}\r\n")
                ]

                summary: MemorySummary = self.llm.with_structured_output(MemorySummary).invoke(messages)
                self.logger.info(f"{summary}")

                self.context = []
            elif decision.action == TranscriptionAction.DISCARD:
                pass

            return decision

