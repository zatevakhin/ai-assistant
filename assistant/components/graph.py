from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing import List, Union
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START

from assistant.components.event_bus import EventBus, EventType
from assistant.components.transcriber import TranscribedSegment
from assistant.components.util import ensure_model_exists
from assistant.config import (
    OLLAMA_LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_KEEP_ALIVE
)


logger = logging.getLogger(__name__)

class SystemIIIContext(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]


class SystemIII:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.voice_query_subscription = self.event_bus.subscribe(EventType.TRANSCRIPTION_DONE, self.on_voice_query)

        self.llm = self.create_llm("deepseek-r1:8b")

        self.running = False
        logger.info(f"{__name__} ... IDLE")
        self.graph = self.__build_graph()

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

    def on_voice_query(self, t: TranscribedSegment):
        logger.info(f"on_voice_query({t})")
        if t.language not in ["en"]:
            logger.warning(f"Language '{t.language}' will not be handled.")
            return

        min_probability = 0.6
        if t.probability < min_probability:
            logger.warning(f"Probability that this is '{t.language}' language is below '{min_probability}'.")
            return


        print("---")
        self.graph.invoke({"messages": [HumanMessage(content=t.text)]})
        print("----")

    def __build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(SystemIIIContext)
        graph.add_node("node_overview", self.node_overview)
        graph.add_conditional_edges("node_overview", self.node_router)
        graph.add_edge(START, "node_overview")

        return graph.compile()

    def node_router(self, ctx: SystemIIIContext):
        logger.info(f"call node_router({ctx})")
        return END


    def node_overview(self, ctx: SystemIIIContext):
        logger.info(f"call node_overview({ctx})")
        print(self.llm.invoke(ctx.messages))

    def run(self):
        self.running = True
        logger.info(f"{__name__} ... OK")

    def stop(self):
        logger.info(f"{__name__} ... STOPPING")
        self.running = False
        self.voice_query_subscription.dispose()
        logger.info(f"{__name__} ... DEAD")

