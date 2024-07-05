from reactivex.subject import Subject
from queue import Queue
import threading


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

