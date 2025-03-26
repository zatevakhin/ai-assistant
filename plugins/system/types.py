from pydantic import BaseModel, Field
from uuid import uuid4, UUID

class TranscribedSegment(BaseModel):
    text: str
    language: str
    probability: float
    uuid: UUID = Field(default_factory=uuid4)

