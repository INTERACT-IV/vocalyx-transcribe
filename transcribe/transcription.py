"""
Fonctions de transcription audio
"""

import asyncio
import concurrent.futures
import gc
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel

from config import Config
from database import SessionLocal, Transcription
from transcribe.audio_utils import get_audio_duration, preprocess_audio, split_audio_intelligent

config = Config()
logger = logging.getLogger(__name__)

# Variables globales
whisper_model = None
executor = None

async def initialize_whisper_model():
    """Initialise le modèle Whisper"""
    global whisper_model, executor
    
    logger.info(f"🚀 Loading Whisper model: {config.model} on {config.device}")
    whisper_model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads
    )
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
    logger.info(f"✅ Whisper loaded | VAD: {config.vad_enabled} | Workers: {config.max_workers}")

async def cleanup_resources():
    """Nettoie les ressources"""
    global whisper_model, executor
    
    try:
        logger.info("🛑 Releasing Whisper model resources...")
        whisper_model = None
        gc.collect()

        if executor:
            logger.info("🧹 Shutting down thread pool executor...")
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
    except Exception as e:
        logger.warning(f"⚠️ Error while cleaning up resources: {e}")

def transcribe_segment(file_path: Path, translate: bool = False) -> tuple:
    """
    Transcrit un segment audio.
    Retourne: (text, segments_list, detected_language)
    """
    global whisper_model
    
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")
    
    segments_list = []
    text_full = ""
    
    segments, info = whisper_model.transcribe(
        str(file_path),
        language=config.language,
        task="translate" if translate else "transcribe",
        beam_size=config.beam_size,
        best_of=config.beam_size,
        temperature=config.temperature,
        vad_filter=True,
        vad_parameters=dict(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            min_silence_duration_ms=config.vad_min_silence_duration_ms
        ),
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    
    for seg in segments:
        segments_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        })
        text_full += seg.text.strip() + " "
    
    return text_full.strip(), segments_list, info.language

async def run_transcription_optimized(
    transcription_id: str,
    file_path: Path,
    translate: bool,
    use_vad: bool = True
):
    """Transcription optimisée avec durée correcte"""
    db = SessionLocal()
    entry = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    if not entry:
        db.close()
        return

    entry.status = "processing"
    entry.vad_enabled = 1 if use_vad else 0
    
    db.commit()
    
    segment_paths = []
    processed_path = None

    try:
        start_time = time.time()
        
        # 1. Obtenir la durée RÉELLE de l'audio original
        original_duration = get_audio_duration(file_path)
        logger.info(f"[{transcription_id}] 📏 Original audio duration: {original_duration}s")
        
        # 2. Pré-traitement audio
        processed_path = preprocess_audio(file_path)
        
        # 3. Découpe intelligente
        segment_paths = split_audio_intelligent(processed_path, use_vad=use_vad)
        logger.info(f"[{transcription_id}] 🔪 Created {len(segment_paths)} segments")

        # 4. Transcription parallèle
        loop = asyncio.get_running_loop()
        
        if len(segment_paths) == 1:
            results = [await loop.run_in_executor(
                executor, transcribe_segment, segment_paths[0], translate
            )]
        else:
            results = await asyncio.gather(*[
                loop.run_in_executor(executor, transcribe_segment, seg, translate)
                for seg in segment_paths
            ])

        # 5. Assemblage des résultats
        full_text = ""
        full_segments = []
        language_detected = None
        time_offset = 0.0

        for text, segments_list, lang in results:
            for seg in segments_list:
                seg["start"] = round(seg["start"] + time_offset, 2)
                seg["end"] = round(seg["end"] + time_offset, 2)
                full_segments.append(seg)
            
            if segments_list:
                time_offset = full_segments[-1]["end"]
            
            full_text += text + " "
            if not language_detected:
                language_detected = lang

        processing_time = round(time.time() - start_time, 3)
        speed_ratio = round(original_duration / processing_time, 2) if processing_time > 0 else 0

        # 6. Mise à jour DB avec durée CORRECTE
        entry.status = "done"
        entry.language = language_detected
        entry.processing_time = processing_time
        entry.duration = original_duration  # ✅ FIX: Utiliser la durée originale
        entry.text = full_text.strip()
        entry.segments = json.dumps(full_segments)
        entry.segments_count = len(full_segments)
        entry.finished_at = datetime.utcnow()
        db.commit()
        
        logger.info(
            f"[{transcription_id}] ✅ Completed: {len(full_segments)} segments | "
            f"Audio: {original_duration:.1f}s | Processing: {processing_time:.1f}s | "
            f"Speed: {speed_ratio}x realtime | VAD: {use_vad}"
        )

    except Exception as e:
        logger.exception(f"[{transcription_id}] ❌ Error: {e}")
        entry.status = "error"
        entry.error_message = str(e)
        entry.finished_at = datetime.utcnow()
        db.commit()

    finally:
        db.close()
        # Cleanup
        try:
            file_path.unlink(missing_ok=True)
            if processed_path and processed_path != file_path:
                processed_path.unlink(missing_ok=True)
            for seg_path in segment_paths:
                seg_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"[{transcription_id}] Cleanup error: {e}")