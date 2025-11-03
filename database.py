"""
Configuration de la base de données et modèles
"""

import uuid
import secrets
import string
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Text, Enum, DateTime, Integer, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from config import Config

config = Config()
logger = logging.getLogger("vocalyx")

Base = declarative_base()

engine = create_engine(config.database_path)
SessionLocal = sessionmaker(bind=engine)

def generate_api_key():
    """Génère une clé d'API sécurisée"""
    alphabet = string.ascii_letters + string.digits
    return 'vk_' + ''.join(secrets.choice(alphabet) for i in range(32))

class Project(Base):
    """Nouveau modèle pour les projets et leurs clés d'API"""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, default=generate_api_key)
    created_at = Column(DateTime, default=datetime.utcnow)

class Transcription(Base):
    """Modèle pour les transcriptions audio"""
    __tablename__ = "transcriptions"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(Enum("pending", "processing", "done", "error", name="transcription_status"), default="pending", index=True)
    
    project_name = Column(String, index=True, nullable=False)
    worker_id = Column(String, nullable=True, index=True)
    file_path = Column(String, nullable=True)

    language = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    text = Column(Text, nullable=True)
    segments = Column(Text, nullable=True)  # JSON
    error_message = Column(Text, nullable=True)
    segments_count = Column(Integer, nullable=True)
    vad_enabled = Column(Integer, default=0)
    
    enrichment_requested = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


def get_or_create_project(db: Session, project_name: str) -> Project:
    """
    Récupère un projet par son nom. S'il n'existe pas,
    il le crée et retourne la nouvelle instance.
    """
    if not project_name:
        raise ValueError("Le nom du projet ne peut pas être vide")

    project = db.query(Project).filter(Project.name == project_name).first()
    
    if project:
        logger.info(f"Projet technique '{project_name}' trouvé.")
        return project
        
    logger.warning(f"Projet technique '{project_name}' non trouvé. Création...")
    
    new_project = Project(name=project_name)
    db.add(new_project)
    try:
        db.commit()
        db.refresh(new_project)
        logger.info(f"✅ Projet '{new_project.name}' créé avec la clé: {new_project.api_key[:6]}...")
        return new_project
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur lors de la création du projet technique: {e}")
        raise