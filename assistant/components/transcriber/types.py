from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TranscribedSegment(BaseModel):
    text: str
    language: str
    probability: float
    uuid: UUID = Field(default_factory=uuid4)
    start_at: Optional[datetime] = None
    end_at: datetime = Field(default_factory=datetime.now)


class Word(BaseModel):
    """Word-level data with timestamps"""

    word: str
    start: float
    end: float


class Speaker(BaseModel):
    """Speaker information from diarization"""

    id: str
    label: str
    total_time: float


class Segment(BaseModel):
    """Segment of transcript with timing and speaker information"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    words: List[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    """Complete transcript with all metadata"""

    transcript: str
    language: str
    duration: float
    speakers: List[Speaker] = Field(default_factory=list)
    segments: List[Segment] = Field(default_factory=list)
