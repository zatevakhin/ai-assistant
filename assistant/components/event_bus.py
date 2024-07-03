from reactivex import Subject
from typing import Dict, Any


class EventBus:
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}

    def get_subject(self, event_type: str) -> Subject:
        if event_type not in self.subjects:
            self.subjects[event_type] = Subject()
        return self.subjects[event_type]

    def publish(self, event_type: str, data: Any):
        subject = self.get_subject(event_type)
        subject.on_next(data)

    def subscribe(self, event_type: str, observer):
        subject = self.get_subject(event_type)
        return subject.subscribe(observer)
