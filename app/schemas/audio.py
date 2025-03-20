from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Word(BaseModel):
    """Represents a single word in the transcription"""

    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: float
    speaker: Optional[int] = None


class Segment(BaseModel):
    """Represents a segment of transcribed audio"""

    start: float
    end: float
    text: str
    words: List[Word]
    speaker: Optional[int] = None


class AudioTranscription(BaseModel):
    """Base audio transcription model"""

    segments: List[Segment]
    word_segments: Optional[List[Dict[str, Any]]] = None


class AudioTranscriptionVerbose(AudioTranscription):
    """Extended audio transcription model with additional details"""

    language: str
    duration: float
    text: str
    words: List[Word]
