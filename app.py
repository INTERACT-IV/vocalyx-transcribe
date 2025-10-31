"""
Point d'entrée principal de l'application FastAPI
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from config import Config
from database import engine, Base, Transcription
from api.endpoints import router as api_router
from api.dependencies import get_db
from logging_config import setup_logging, get_uvicorn_log_config

# Initialiser la configuration
config = Config()

# Configurer le logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="logs/vocalyx.log" if not os.getenv("NO_LOG_FILE") else None
)

# Créer les tables
Base.metadata.create_all(bind=engine)

# Initialiser le limiteur de taux
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("🚀 Démarrage de l'application Vocalyx")
    
    # Import depuis le nouveau module
    from transcribe.transcription import initialize_whisper_model
    await initialize_whisper_model()
    
    yield  # --- App runs here ---
    
    # --- Shutdown ---
    logger.info("🛑 Arrêt de l'application Vocalyx")
    from transcribe.transcription import cleanup_resources
    await cleanup_resources()

# Créer l'application FastAPI
app = FastAPI(
    title="Vocalyx API",
    description="Vocalyx transforme automatiquement les enregistrements de call centers en transcriptions enrichies et exploitables.",
    version="1.4.0",  # Bump version
    contact={"name": "Guilhem RICHARD", "email": "guilhem.l.richard@gmail.com"},
    lifespan=lifespan
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Configurer le limiteur de taux
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Inclure les routes de l'API
app.include_router(api_router, prefix="/api")

# Configurer les templates
templates = Jinja2Templates(directory=config.templates_dir)

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root(request: Request):
    """Page d'accueil - redirige vers le dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard(request: Request, limit: int = 10, db: Session = Depends(get_db)):
    entries = db.query(Transcription).order_by(Transcription.created_at.desc()).limit(limit).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "entries": entries})

if __name__ == "__main__":
    log_config = get_uvicorn_log_config(
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=log_config
    )