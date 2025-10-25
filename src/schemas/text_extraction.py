from pydantic import BaseModel, Field


class TextExtractionResult(BaseModel):
    text: str
    character_length: int
    token_length: int
    duration: float = Field(description="duration in seconds")
