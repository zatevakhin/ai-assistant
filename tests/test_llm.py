from queue import Queue
from time import sleep
import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistant.components import LlmInferenceProcess, EventBus
from assistant.config import TOPIC_TRANSCRIPTION_DONE, TOPIC_LLM_ON_SENTENCE

@pytest.fixture(scope="module")
def event_bus():
    eb = EventBus()
    yield eb

@pytest.fixture(scope="module")
def llm(event_bus: EventBus):
    process = LlmInferenceProcess(event_bus)

    process.run()
    yield process
    process.stop()

# TODO: Add test to ensure that LLM service is available.
# TODO: Add test to ensure that specific LLM model is available.

def test_is_inference_working(llm: LlmInferenceProcess, event_bus: EventBus):
    answer = Queue()
    event_bus.subscribe(TOPIC_LLM_ON_SENTENCE, answer.put)

    event_bus.publish(TOPIC_TRANSCRIPTION_DONE, {
        "text": "Hello!",
        "language": "en",
        "probability": 1
    })

    sleep(5)

    assert len(answer.get()) > 0
    answer.task_done()

    while not answer.empty():
        answer.get()
        answer.task_done()

    answer = Queue()
    event_bus.subscribe(TOPIC_LLM_ON_SENTENCE, answer.put)

    event_bus.publish(TOPIC_TRANSCRIPTION_DONE, {
        "text": "Why is the sky blue?",
        "language": "en",
        "probability": 1
    })

    sleep(5)

    assert len(answer.get()) > 0
    answer.task_done()

    while not answer.empty():
        answer.get()
        answer.task_done()
