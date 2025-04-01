import logging
import threading
from contextlib import contextmanager
from queue import Queue
from typing import Callable, Optional

from ollama import Client
from reactivex.subject import Subject
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


def observe(q: Queue, fn: Callable, threaded: bool = False, max_workers: Optional[int] = None) -> Subject:
    subject = Subject()
    if threaded:
        executor = ThreadPoolExecutor(max_workers=max_workers)

        subject.subscribe(
            on_next=lambda item: executor.submit(fn, item),
            on_completed=lambda: executor.shutdown(wait=False),
        )
    else:
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


def title_to_snake(s: str) -> str:
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()
