from reactivex.subject import Subject
from typing import Callable
from queue import Queue
import threading
from contextlib import contextmanager
import logging
from ollama import Client
from tqdm import tqdm

logger = logging.getLogger(__name__)


def observe(q: Queue, fn: Callable) -> Subject:
    subject = Subject()
    subject.subscribe(fn)

    def producer():
        while not subject.is_disposed:
            item = q.get()
            if item is None:
                subject.on_completed()
                q.task_done()
                break
            subject.on_next(item)
            q.task_done()

    threading.Thread(target=producer, daemon=True).start()
    return subject


@contextmanager
def event_context(e: threading.Event):
    try:
        e.set()
        yield e
    finally:
        e.clear()


def ensure_model_exists(base_url: str, model: str):
    o_client = Client(base_url)
    models = o_client.list().get("models")

    if model not in map(lambda item: item["model"], models):
        logger.warning(f"Model '{model}' does not exists. Downloading.")

        for _ in tqdm(o_client.pull(model, stream=True)):
            pass
