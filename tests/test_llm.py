from queue import Queue
from time import sleep
import pytest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistant.components import (
    LlmInferenceProcess,
    EventBus,
    EventType,
    TranscribedSegment,
)


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
# TODO: Add history reset and add test for it.


def test_is_inference_working(llm: LlmInferenceProcess, event_bus: EventBus):
    answer = Queue()
    event_bus.subscribe(EventType.LLM_NEW_SENTENCE, answer.put)

    event_bus.publish(
        EventType.TRANSCRIPTION_DONE,
        TranscribedSegment(text="Hello!", language="en", probability=1),
    )

    sleep(5)

    assert len(answer.get()) > 0
    answer.task_done()

    while not answer.empty():
        answer.get()
        answer.task_done()

    answer = Queue()
    event_bus.subscribe(EventType.LLM_NEW_SENTENCE, answer.put)
    event_bus.publish(
        EventType.TRANSCRIPTION_DONE,
        TranscribedSegment(text="Why is the sky blue?", language="en", probability=1),
    )

    sleep(5)

    assert len(answer.get()) > 0
    answer.task_done()

    while not answer.empty():
        answer.get()
        answer.task_done()


def test_interruption(llm: LlmInferenceProcess, event_bus: EventBus):
    answer = Queue()
    discarded = Queue()
    event_bus.subscribe(EventType.LLM_NEW_SENTENCE, answer.put)
    event_bus.subscribe(EventType.LLM_SENTENCE_DISCARD, discarded.put)

    event_bus.publish(
        EventType.TRANSCRIPTION_DONE,
        TranscribedSegment(
            text="Hello! Could you explain thoroughly what are mitochondrias?",
            language="en",
            probability=1,
        ),
    )

    event_bus.publish(
        EventType.TRANSCRIPTION_DONE,
        TranscribedSegment(text="Why is the sky blue?", language="en", probability=1),
    )

    sleep(2)

    while not answer.empty():
        print(answer.get())
        answer.task_done()

    # NOTE: Tests is broken for now.
    assert not discarded.empty()
