from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TranscribedSegment(BaseModel):
    text: str
    language: str
    probability: float
    uuid: UUID = Field(default_factory=uuid4)
