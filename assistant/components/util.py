from reactivex.subject import Subject
from typing import Callable, Any, Optional
from queue import Queue
import threading
import time
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

def queue_as_observable(q: Queue) -> Subject:
    subject = Subject()

    def producer():
        while True:
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
def controlled_area(callback: Callable, in_event: Any, out_event: Any, measure_time: bool = False, scope: Optional[str] = None):
    callback(in_event)

    t_start: Optional[float] = None
    if measure_time:
        t_start = time.perf_counter()

    yield

    if measure_time:
        t_end = time.perf_counter()
        scope = f"[{scope}] " if scope is not None else ""

        logger.info(f"{scope}Execution time: {t_end - t_start:0.9f} seconds")


    callback(out_event)


def ensure_model_exists(base_url: str, model: str):
    o_client = Client(base_url)
    models = o_client.list().get("models")

    if model not in map(lambda item: item["model"], models):
        logger.warning(f"Model '{model}' does not exists. Downloading.")

        for _ in tqdm(o_client.pull(model, stream=True)):
            pass

