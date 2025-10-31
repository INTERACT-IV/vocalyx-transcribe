"""
Points de terminaison API
"""

import json
import logging
import platform
import psutil
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from config import Config
from database import Transcription
from models.schemas import TranscriptionResult
from api.dependencies import get_db

config = Config()
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=config.templates_dir)

router = APIRouter()

@router.get("/transcribe/count", tags=["Transcriptions"])
def get_transcription_count(db: Session = Depends(get_db)):
    """
    Retourne le nombre total de transcriptions et leur répartition par statut.
    """
    from sqlalchemy import func
    
    counts = (
        db.query(Transcription.status, func.count(Transcription.id))
        .group_by(Transcription.status)
        .all()
    )

    result = {"total": 0, "pending": 0, "processing": 0, "done": 0, "error": 0}
    for status, count in counts:
        if status in result:
            result[status] = count
            result["total"] += count

    return result

@router.get("/transcribe/recent", response_model=List[TranscriptionResult], tags=["Transcriptions"])
def get_recent_transcriptions(
    limit: int = Query(10, ge=1, le=100), 
    page: int = Query(1, ge=1),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Récupère les transcriptions récentes avec filtres et pagination.
    """
    query = db.query(Transcription)
    
    if status:
        query = query.filter(Transcription.status == status)
    if search:
        query = query.filter(Transcription.text.ilike(f"%{search}%"))
        
    entries = query.order_by(Transcription.created_at.desc()).limit(limit).offset((page - 1) * limit).all()
    
    results = []
    for entry in entries:
        segments = json.loads(entry.segments) if entry.segments else []
        results.append({
            "id": entry.id,
            "status": entry.status,
            "project_name": entry.project_name,
            "worker_id": entry.worker_id,
            "language": entry.language,
            "processing_time": float(entry.processing_time) if entry.processing_time else None,
            "duration": float(entry.duration) if entry.duration else None,
            "text": entry.text,
            "segments": segments,
            "error_message": entry.error_message,
            "segments_count": entry.segments_count,
            "vad_enabled": bool(entry.vad_enabled),
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "finished_at": entry.finished_at.isoformat() if entry.finished_at else None,
        })
    return results

@router.get("/transcribe/{transcription_id}", response_model=TranscriptionResult, tags=["Transcriptions"])
def get_transcription(transcription_id: str, db: Session = Depends(get_db)):
    entry = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    if not entry:
        raise HTTPException(404, "Not found")
    
    segments = json.loads(entry.segments) if entry.segments else []
    return {
        "id": entry.id,
        "status": entry.status,
        "project_name": entry.project_name,
        "worker_id": entry.worker_id,
        "language": entry.language,
        "processing_time": float(entry.processing_time) if entry.processing_time else None,
        "duration": float(entry.duration) if entry.duration else None,
        "text": entry.text,
        "segments": segments,
        "error_message": entry.error_message,
        "segments_count": entry.segments_count,
        "vad_enabled": bool(entry.vad_enabled),
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "finished_at": entry.finished_at.isoformat() if entry.finished_at else None,
    }

@router.delete("/transcribe/{transcription_id}", tags=["Transcriptions"])
def delete_transcription(transcription_id: str, db: Session = Depends(get_db)):
    entry = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    if not entry:
        raise HTTPException(404, "Not found")
    db.delete(entry)
    db.commit()
    return {"status": "deleted", "id": transcription_id}

@router.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard(request: Request, limit: int = 10, db: Session = Depends(get_db)):
    entries = db.query(Transcription).order_by(Transcription.created_at.desc()).limit(limit).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "entries": entries})

@router.get("/config", tags=["System"])
def get_config():
    """Retourne la configuration actuelle (sans données sensibles)"""
    return {
        "core": {
            "instance_name": config.instance_name
        },
        "whisper": {
            "model": config.model,
            "device": config.device,
            "compute_type": config.compute_type,
            "language": config.language,
        },
        "performance": {
            "max_workers": config.max_workers,
            "segment_length_ms": config.segment_length_ms,
            "vad_enabled": config.vad_enabled,
        },
        "limits": {
            "max_file_size_mb": config.max_file_size_mb,
            "rate_limit_per_minute": config.rate_limit,
            "allowed_extensions": list(config.allowed_extensions),
        }
    }

@router.post("/config/reload", tags=["System"])
def reload_config():
    """Recharge la configuration depuis le fichier"""
    try:
        config.reload()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(500, f"Failed to reload config: {str(e)}")

@router.get("/health", tags=["System"])
def health_check(request: Request):
    """Modifié pour utiliser app.state"""
    service = request.app.state.transcription_service
    model_loaded = service.model is not None
    
    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat(),
        "config_file": "config.ini"
    }

@router.get("/worker/status", tags=["System"])
def get_worker_status(request: Request):
    """🆕 Nouveau endpoint pour le monitoring"""
    service = request.app.state.transcription_service
    config = request.app.state.config

    usage_percent = 0
    if config.max_workers > 0:
        usage_percent = round((service.active_tasks / config.max_workers) * 100, 1)
        
    # Calcul Uptime
    uptime_seconds = (datetime.utcnow() - service.start_time).total_seconds()
    
    return {
        "instance_name": config.instance_name,
        "status": "idle" if service.active_tasks == 0 else "processing",
        "max_workers": config.max_workers,
        "active_tasks": service.active_tasks,
        "usage_percent": usage_percent,
        
        # 🆕 Stats machine
        "machine_name": platform.node(),
        "cpu_usage_percent": psutil.cpu_percent(),
        "memory_usage_percent": psutil.virtual_memory().percent,
        
        # 🆕 Stats cumulatives
        "start_time_utc": service.start_time.isoformat(),
        "uptime_seconds": uptime_seconds,
        "total_jobs_completed": service.total_jobs_completed,
        "total_audio_processed_s": round(service.total_audio_processed_s, 2),
        "total_processing_time_s": round(service.total_processing_time_s, 2),
    }