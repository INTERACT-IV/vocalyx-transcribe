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
from sqlalchemy.orm import Session

from config import Config
from database import SessionLocal, Transcription
from transcribe.audio_utils import get_audio_duration, preprocess_audio, split_audio_intelligent

logger = logging.getLogger(__name__)

class TranscriptionService:
    """
    Service de transcription encapsulant le modèle Whisper et le pool d'exécution.
    Ceci évite les variables globales.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.active_tasks = 0

        self.start_time = datetime.utcnow()
        self.total_jobs_completed = 0
        self.total_audio_processed_s = 0.0
        self.total_processing_time_s = 0.0

        self._load_model()
        
    def _load_model(self):
        logger.info(f"🚀 Loading Whisper model: {self.config.model} on {self.config.device}")
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )
        logger.info(f"✅ Whisper loaded | VAD: {self.config.vad_enabled} | Workers: {self.config.max_workers}")

    async def cleanup_resources(self):
        """Nettoie les ressources"""
        try:
            logger.info("🛑 Releasing Whisper model resources...")
            del self.model
            self.model = None
            gc.collect()

            if self.executor:
                logger.info("🧹 Shutting down thread pool executor...")
                self.executor.shutdown(wait=False)
                self.executor = None
        except Exception as e:
            logger.warning(f"⚠️ Error while cleaning up resources: {e}")

    def transcribe_segment(self, file_path: Path, translate: bool = False) -> tuple:
        """
        Transcrit un segment audio.
        Retourne: (text, segments_list, detected_language)
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        segments_list = []
        text_full = ""
        
        segments, info = self.model.transcribe(
            str(file_path),
            language=self.config.language,
            task="translate" if translate else "transcribe",
            beam_size=self.config.beam_size,
            best_of=self.config.beam_size,
            temperature=self.config.temperature,
            vad_filter=True,
            vad_parameters=dict(
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.config.vad_min_silence_duration_ms
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
        self,
        transcription_id: str,
        file_path_str: str,
        worker_id: str
    ):
        """Transcription optimisée avec durée correcte"""
        
        self.active_tasks += 1
        db = SessionLocal()
        entry = db.query(Transcription).filter(Transcription.id == transcription_id).first()
        
        if not entry:
            db.close()
            self.active_tasks -= 1
            return
            
        file_path = Path(file_path_str)
        
        # Mettre à jour le statut (déjà 'processing' mais re-confirmer)
        entry.worker_id = worker_id
        entry.status = "processing"
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
            use_vad = bool(entry.vad_enabled)
            segment_paths = split_audio_intelligent(processed_path, use_vad=use_vad)
            logger.info(f"[{transcription_id}] 🔪 Created {len(segment_paths)} segments")

            # 4. Transcription parallèle
            loop = asyncio.get_running_loop()
            translate = False # (A-t-on besoin de ça ?) On le garde simple pour l'instant.
            
            if len(segment_paths) == 1:
                results = [await loop.run_in_executor(
                    self.executor, self.transcribe_segment, segment_paths[0], translate
                )]
            else:
                results = await asyncio.gather(*[
                    loop.run_in_executor(self.executor, self.transcribe_segment, seg, translate)
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
            entry.duration = original_duration
            entry.text = full_text.strip()
            entry.segments = json.dumps(full_segments)
            entry.segments_count = len(full_segments)
            entry.finished_at = datetime.utcnow()
            db.commit()


            # 7. Statistiques globales
            self.total_jobs_completed += 1
            self.total_audio_processed_s += original_duration
            self.total_processing_time_s += processing_time
            
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
            self.active_tasks -= 1
            db.close()
            # Cleanup
            try:
                # Ne pas supprimer le fichier original, seulement les fichiers traités
                # file_path.unlink(missing_ok=True) # Fait par le dashboard ?
                if processed_path and processed_path != file_path:
                    processed_path.unlink(missing_ok=True)
                for seg_path in segment_paths:
                    seg_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"[{transcription_id}] Cleanup error: {e}")