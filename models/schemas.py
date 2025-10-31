"""
Schémas Pydantic pour les modèles de données
"""

from typing import Optional, List
from pydantic import BaseModel

class TranscriptionResult(BaseModel):
    id: str
    status: str
    language: Optional[str] = None
    processing_time: Optional[float] = None
    duration: Optional[float] = None
    text: Optional[str] = None
    segments: Optional[list] = None
    error_message: Optional[str] = None
    segments_count: Optional[int] = None
    vad_enabled: Optional[bool] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None