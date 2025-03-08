from reactivex import Subject
from typing import Dict, Any
from enum import Enum, auto

class EventType(Enum):
    MUMBLE_NEW_AUDIO = auto()
    MUMBLE_PLAY_AUDIO = auto()
    MUMBLE_NOW_PLAYING = auto()
    MUMBLE_INTERRUPT_AUDIO = auto()
    MUMBLE_PLAYING_STATUS = auto()

    VAD_NEW_SPEECH = auto()

    TRANSCRIPTION_DONE = auto()
    TRANSCRIPTION_STATUS = auto()

    LLM_STREAM_DONE = auto()
    LLM_NEW_SENTENCE = auto()
    LLM_SENTENCE_DISCARD = auto()
    LLM_STREAM_INTERRUPT = auto()
    LLM_INFERENCE_STATUS = auto()

    SPEECH_SYNTH_STATUS = auto()


class EventBus:
    def __init__(self):
        self.subjects: Dict[EventType, Subject] = {}

    def get_subject(self, event_type: EventType) -> Subject:
        if event_type not in self.subjects:
            self.subjects[event_type] = Subject()
        return self.subjects[event_type]

    def publish(self, event_type: EventType, data: Any):
        subject = self.get_subject(event_type)
        subject.on_next(data)

    def subscribe(self, event_type: EventType, observer):
        subject = self.get_subject(event_type)
        return subject.subscribe(observer)
