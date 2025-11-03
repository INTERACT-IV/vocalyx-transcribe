"""
Dépendances FastAPI
"""

import secrets
import logging
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from database import SessionLocal
from config import Config

# ❗️ AJOUT: Charger la config pour lire la clé
config = Config()
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ❗️ NOUVEAU: Dépendance pour sécuriser les endpoints internes
def verify_internal_access(
    x_internal_api_key: str = Header(None, description="Clé d'API interne pour la communication inter-services")
):
    """
    Valide que la requête provient d'un service interne (ex: le Dashboard)
    """
    if not config.internal_api_key:
        # Si la clé n'est pas configurée sur le worker, on log une erreur et on refuse tout
        logger.critical("❌ Sécurité: Endpoint interne appelé, mais 'internal_api_key' n'est pas configurée!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Internal service not configured"
        )
        
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, config.internal_api_key):
        logger.warning(f"🚫 Accès interne refusé (Clé invalide: {x_internal_api_key[:4]}...)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid internal API key"
        )
    
    return True