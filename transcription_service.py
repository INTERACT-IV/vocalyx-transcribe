"""
vocalyx-transcribe/transcription_service.py
Service de transcription avec Whisper (adapté pour worker Celery)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from faster_whisper import WhisperModel
from audio_utils import get_audio_duration, preprocess_audio, split_audio_intelligent
from timeout_utils import timeout, TimeoutError

import signal
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TranscriptionService:
    """
    Service de transcription encapsulant le modèle Whisper.
    Version simplifiée pour worker Celery (pas de pool d'exécution, Celery gère la concurrence).
    """
    
    def __init__(self, config):
        self.config = config
        
        # Statistiques du worker
        self.total_jobs_completed = 0
        self.total_audio_processed_s = 0.0
        self.total_processing_time_s = 0.0
        
        # Charger le modèle Whisper
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle Whisper"""
        logger.info(f"🚀 Loading Whisper model: {self.config.model}")
        logger.info(f"📊 Device: {self.config.device} | Compute: {self.config.compute_type}")
        
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads
        )
        
        logger.info(f"✅ Whisper model loaded successfully")
        logger.info(f"⚙️ VAD: {self.config.vad_enabled} | Beam size: {self.config.beam_size}")
        
    def transcribe_segment(self, file_path: Path, use_vad: bool = True, retry_without_vad: bool = True) -> Tuple[str, List[Dict], str]:
        """
        Transcrit un segment audio avec consommation progressive du générateur.
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        segments_list = []
        text_full = ""
        
        logger.info(f"🎯 Starting Whisper transcription (VAD: {use_vad})...")
        
        try:
            vad_params = None
            if use_vad:
                vad_params = {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "max_speech_duration_s": float('inf'),
                    "min_silence_duration_ms": 2000,
                    "speech_pad_ms": 400
                }
            
            segments, info = self.model.transcribe(
                str(file_path),
                language=self.config.language or None,
                task="transcribe",
                beam_size=self.config.beam_size,
                best_of=self.config.beam_size,
                temperature=self.config.temperature,
                vad_filter=use_vad,
                vad_parameters=vad_params,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            
            logger.info(f"🎯 Whisper inference completed, consuming generator...")
            
            # ✅ NOUVELLE APPROCHE : Consommer progressivement avec timeout global
            segments_list_raw = []
            start_consume_time = time.time()
            timeout_seconds = 300  # 5 minutes max
            segment_count = 0
            
            logger.info(f"📝 Consuming generator progressively (timeout: {timeout_seconds}s)...")
            logger.info(f"⏱Segments {segments}")
            logger.info(f"⏱lenSegments {len(segments)}")
            logger.info(f"⏱Info {info}")
            for seg in segments:
                # Vérifier le timeout à chaque itération
                elapsed = time.time() - start_consume_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Generator consumption exceeded {timeout_seconds}s")
                
                segments_list_raw.append(seg)
                segment_count += 1
                
                # Log tous les 5 segments
                if segment_count % 5 == 0:
                    logger.info(f"📝 Consumed {segment_count} segments (elapsed: {elapsed:.1f}s)")
            
            consume_time = time.time() - start_consume_time
            logger.info(f"✅ Generator consumed, got {len(segments_list_raw)} segments in {consume_time:.1f}s")
            
        except TimeoutError as e:
            logger.error(f"❌ Timeout while consuming generator: {e}")
            
            if use_vad and retry_without_vad:
                logger.warning(f"⚠️ Retrying WITHOUT VAD...")
                return self.transcribe_segment(file_path, use_vad=False, retry_without_vad=False)
            else:
                raise Exception(f"Generator consumption timed out: {e}")
        
        except StopIteration:
            logger.info(f"✅ Generator exhausted normally")
        
        except Exception as e:
            logger.error(f"❌ Error during transcription: {e}", exc_info=True)
            raise
        
        # Convertir les segments en dictionnaires
        logger.info(f"📝 Converting {len(segments_list_raw)} segments to dict...")
        
        for i, seg in enumerate(segments_list_raw):
            try:
                segments_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip()
                })
                text_full += seg.text.strip() + " "
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📝 Processed {i + 1}/{len(segments_list_raw)} segments")
                    
            except Exception as e:
                logger.error(f"❌ Error processing segment {i}: {e}")
                continue
        
        logger.info(f"✅ All {len(segments_list)} segments processed")
        
        return text_full.strip(), segments_list, info.language
    
    def transcribe(self, file_path: str, use_vad: bool = True) -> Dict:
        """
        Transcrit un fichier audio (point d'entrée principal).
        
        Args:
            file_path: Chemin vers le fichier audio
            use_vad: Utiliser la détection de voix
            
        Returns:
            dict: Résultats de la transcription
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"📁 Processing file: {file_path.name} | VAD requested: {use_vad}")
        
        segment_paths = []
        processed_path = None
        
        try:
            start_time = time.time()
            
            # 1. Obtenir la durée réelle de l'audio
            original_duration = get_audio_duration(file_path)
            logger.info(f"📏 Audio duration: {original_duration}s")
            
            # 2. Pré-traitement audio (normalisation, conversion mono 16kHz)
            processed_path = preprocess_audio(file_path)
            logger.info(f"✨ Audio preprocessed")
            
            # 3. Découpe intelligente (si nécessaire)
            segment_paths = split_audio_intelligent(
                processed_path,
                use_vad=use_vad  # ← Passer le paramètre correct
            )
            logger.info(f"🔪 Created {len(segment_paths)} segment(s)")
            
            # 4. Transcription
            full_text = ""
            full_segments = []
            language_detected = None
            time_offset = 0.0
            
            for i, segment_path in enumerate(segment_paths):
                logger.info(f"🎤 Transcribing segment {i+1}/{len(segment_paths)}...")
                
                # ✅ CORRECTION : Passer le paramètre use_vad
                text, segments_list, lang = self.transcribe_segment(segment_path, use_vad=use_vad)
                
                # Ajuster les timestamps avec l'offset
                for seg in segments_list:
                    seg["start"] = round(seg["start"] + time_offset, 2)
                    seg["end"] = round(seg["end"] + time_offset, 2)
                    full_segments.append(seg)
                
                # Mettre à jour l'offset pour le prochain segment
                if full_segments:
                    time_offset = full_segments[-1]["end"]
                
                full_text += text + " "
                
                if not language_detected:
                    language_detected = lang
            
            processing_time = round(time.time() - start_time, 2)
            speed_ratio = round(original_duration / processing_time, 2) if processing_time > 0 else 0
            
            logger.info(
                f"✅ Transcription completed | "
                f"Segments: {len(full_segments)} | "
                f"Speed: {speed_ratio}x realtime"
            )
            
            return {
                "text": full_text.strip(),
                "segments": full_segments,
                "language": language_detected,
                "duration": original_duration,
                "processing_time": processing_time,
                "segments_count": len(full_segments)
            }
            
        finally:
            # Nettoyage des fichiers temporaires
            try:
                if processed_path and processed_path != file_path and processed_path.exists():
                    processed_path.unlink()
                
                for seg_path in segment_paths:
                    if seg_path.exists():
                        seg_path.unlink()
                
                logger.debug("🧹 Temporary files cleaned")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup error: {e}")
    
    def update_stats(self, audio_duration: float, processing_time: float):
        """Met à jour les statistiques du worker"""
        self.total_jobs_completed += 1
        self.total_audio_processed_s += audio_duration
        self.total_processing_time_s += processing_time
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du worker"""
        return {
            "total_jobs_completed": self.total_jobs_completed,
            "total_audio_processed_s": round(self.total_audio_processed_s, 2),
            "total_processing_time_s": round(self.total_processing_time_s, 2),
            "avg_speed_ratio": round(
                self.total_audio_processed_s / self.total_processing_time_s, 2
            ) if self.total_processing_time_s > 0 else 0
        }