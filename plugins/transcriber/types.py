from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4, UUID
from datetime import datetime

class TranscribedSegment(BaseModel):
    text: str
    language: str
    probability: float
    uuid: UUID = Field(default_factory=uuid4)
    start_at: Optional[datetime] = None
    end_at: datetime = Field(default_factory=datetime.now)

