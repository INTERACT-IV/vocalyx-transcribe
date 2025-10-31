"""
Points de terminaison API
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from config import Config
from database import Transcription, SessionLocal
from models.schemas import TranscriptionResult
from api.dependencies import get_db

# Import depuis le nouveau module transcribe
from transcribe.transcription import run_transcription_optimized
from transcribe.audio_utils import sanitize_filename

config = Config()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
templates = Jinja2Templates(directory=config.templates_dir)

router = APIRouter()

@router.post("/transcribe", summary="Créer une transcription", tags=["Transcriptions"])
@limiter.limit(f"{config.rate_limit}/minute")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    translate: Optional[bool] = False,
    use_vad: Optional[bool] = True
):
    from transcribe.transcription import whisper_model
    
    if whisper_model is None:
        raise HTTPException(503, "Service starting up. Please wait.")
    
    filename = sanitize_filename(file.filename or "upload")
    ext = filename.split('.')[-1].lower()
    
    if ext not in config.allowed_extensions:
        raise HTTPException(400, f"Unsupported: {ext}")

    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > config.max_file_size_mb:
        raise HTTPException(413, f"File too large: {file_size_mb:.2f}MB")

    transcription_id = str(uuid.uuid4())
    tmp_path = config.upload_dir / f"{transcription_id}_{filename}"
    
    with open(tmp_path, "wb") as fh:
        fh.write(content)

    db = SessionLocal()
    db.add(Transcription(
        id=transcription_id,
        status="pending",
        vad_enabled=1 if use_vad else 0,
        enrichment_requested=1,  # Par défaut, on enrichit
        created_at=datetime.utcnow()
    ))
    db.commit()
    db.close()

    logger.info(f"[{transcription_id}] 📥 {filename} ({file_size_mb:.2f}MB) | VAD: {use_vad}")
    
    import asyncio
    asyncio.create_task(run_transcription_optimized(transcription_id, tmp_path, translate, use_vad))

    return {"transcription_id": transcription_id, "status": "pending"}

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
        result[status] = count
        result["total"] += count

    return result

@router.get("/transcribe/recent", response_model=List[TranscriptionResult], tags=["Transcriptions"])
def get_recent_transcriptions(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    entries = db.query(Transcription).order_by(Transcription.created_at.desc()).limit(limit).all()
    
    results = []
    for entry in entries:
        segments = json.loads(entry.segments) if entry.segments else []
        results.append({
            "id": entry.id,
            "status": entry.status,
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
def health_check():
    from transcribe.transcription import whisper_model
    
    return {
        "status": "healthy" if whisper_model else "starting",
        "model_loaded": whisper_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "config_file": "config.ini"
    }