"""
Point d'entrée principal de l'application FastAPI
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional 
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

import uvicorn

from config import Config
from database import engine, Base, Transcription, SessionLocal
from api.endpoints import router as api_router
from api.dependencies import get_db
# ❗️ MODIFICATION: Import de setup_colored_logging
from logging_config import setup_logging, get_uvicorn_log_config, setup_colored_logging
from transcribe.transcription import TranscriptionService 

# Initialiser la configuration
config = Config()

# ❗️ MODIFICATION: Logique de logging conditionnelle
if config.log_colored:
    logger = setup_colored_logging(
        log_level=config.log_level,
        log_file=config.log_file_path if config.log_file_enabled else None
    )
else:
    logger = setup_logging(
        log_level=config.log_level,
        log_file=config.log_file_path if config.log_file_enabled else None
    )

# Créer les tables
Base.metadata.create_all(bind=engine)

def find_and_lock_job(db: Session, worker_id: str) -> Optional[Transcription]:
    """
    Tente de trouver et de verrouiller atomiquement un job 'pending'.
    Utilise 'SELECT FOR UPDATE SKIP LOCKED' pour la robustesse.
    """
    try:
        # Utilise une transaction pour le verrouillage
        with db.begin_nested():
            # Tente de verrouiller la première tâche en attente (la plus ancienne)
            stmt = select(Transcription)\
                .where(Transcription.status == 'pending')\
                .order_by(Transcription.created_at.asc())\
                .with_for_update(skip_locked=True)
            
            # .scalars().first() récupère le premier objet, ou None, sans erreur
            job = db.execute(stmt).scalars().first()
            
            if job:
                job.status = 'processing'
                job.worker_id = worker_id
                db.commit()
                return job
    except SQLAlchemyError as e:
        logger.error(f"Error locking job: {e}")
        db.rollback()
    
    return None

async def worker_loop(app: FastAPI):
    """
    Boucle de fond qui interroge la DB pour les tâches 'pending'.
    """
    service: TranscriptionService = app.state.transcription_service
    config: Config = app.state.config
    
    logger.info(f"[{config.instance_name}] 👷 Worker loop starting...")
    
    while True:
        await asyncio.sleep(5) # Intervalle de polling
        
        # Ne pas prendre de nouvelles tâches si le pool est plein
        if service.active_tasks >= config.max_workers:
            continue
            
        db = SessionLocal()
        job = None
        try:
            job = find_and_lock_job(db, config.instance_name)
            
            if job:
                logger.info(f"[{config.instance_name}] 📬 Picked up job {job.id}")
                # Exécute la tâche en arrière-plan sans bloquer la boucle
                asyncio.create_task(
                    service.run_transcription_optimized(
                        job.id, 
                        job.file_path, 
                        config.instance_name
                    )
                )
        except Exception as e:
            logger.error(f"[{config.instance_name}] Error in worker loop: {e}")
            if job: # Si l'erreur s'est produite après avoir verrouillé
                job.status = 'error'
                job.error_message = f"Worker loop error: {str(e)}"
                db.commit()
        finally:
            db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("🚀 Démarrage de l'application Vocalyx (Worker)")
    
    # Initialiser le service et le stocker dans app.state
    service = TranscriptionService(config)
    app.state.transcription_service = service
    app.state.config = config
    
    # Démarrer la boucle du worker
    app.state.worker_task = asyncio.create_task(worker_loop(app))
    
    yield  # --- App runs here ---
    
    # --- Shutdown ---
    logger.info("🛑 Arrêt de l'application Vocalyx (Worker)")
    if app.state.worker_task:
        app.state.worker_task.cancel()
    if app.state.transcription_service:
        await app.state.transcription_service.cleanup_resources()

# Créer l'application FastAPI
app = FastAPI(
    title="Vocalyx API (Worker)",
    description="Instance de worker pour la transcription.",
    version="1.0.0",
    contact={"name": "Guilhem RICHARD", "email": "guilhem.l.richard@gmail.com"},
    lifespan=lifespan
)

# Monter les fichiers statiques (toujours utile pour l'accès aux /docs)
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Inclure les routes de l'API
app.include_router(api_router, prefix="/api")

# Configurer les templates
templates = Jinja2Templates(directory=config.templates_dir)

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root(request: Request):
    """Page d'accueil - redirige vers les docs API"""
    return HTMLResponse('<html><body><h1>Vocalyx Worker</h1><a href="/docs">API Docs</a></body></html>')

# La route /dashboard n'est plus la route principale de ce service
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"], dependencies=[Depends(get_db)])
def dashboard(request: Request, limit: int = 10, db: Session = Depends(get_db)):
    entries = db.query(Transcription).order_by(Transcription.created_at.desc()).limit(limit).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "entries": entries})

if __name__ == "__main__":
    # ❗️ MODIFICATION: Utilisation de config.log_level
    log_config = get_uvicorn_log_config(
        log_level=config.log_level 
    )
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_config=log_config
    )