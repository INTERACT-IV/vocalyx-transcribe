"""
Configuration de la base de données et modèles
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Text, Enum, DateTime, Integer, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

from config import Config

config = Config()

Base = declarative_base()
engine = create_engine(config.database_path, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Transcription(Base):
    """Modèle pour les transcriptions audio"""
    __tablename__ = "transcriptions"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(Enum("pending", "processing", "done", "error"), default="pending")
    language = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    text = Column(Text, nullable=True)
    segments = Column(Text, nullable=True)  # JSON
    error_message = Column(Text, nullable=True)
    segments_count = Column(Integer, nullable=True)
    vad_enabled = Column(Integer, default=0)
    
    # 🆕 Pour l'enrichissement
    enrichment_requested = Column(Integer, default=1)  # 1 = oui, 0 = non
    
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)