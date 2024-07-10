from reactivex.subject import Subject
from typing import Callable, Any, Optional
from queue import Queue
import threading
import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

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
