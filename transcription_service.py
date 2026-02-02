"""
vocalyx-transcribe/transcription_service.py
Service de transcription avec Whisper (adapté pour worker Celery)
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from faster_whisper import WhisperModel
from audio_utils import get_audio_duration, preprocess_audio, split_audio_intelligent
from stereo_diarization import StereoDiarizationService
from pyannote_diarization import PyannoteDiarizationService

# Imports non utilisés (TimeoutError, signal, contextmanager) supprimés
logger = logging.getLogger("vocalyx")

class TranscriptionService:
    """
    Service de transcription encapsulant le modèle Whisper.
    Version simplifiée pour worker Celery (pas de pool d'exécution, Celery gère la concurrence).
    """
    
    def __init__(self, config, model_name: Optional[str] = None):
        """
        Initialise le service de transcription.
        Les modèles sont chargés en lazy loading (quand nécessaire).
        
        Args:
            config: Configuration du worker
            model_name: Nom du modèle Whisper à utiliser (tiny, base, small, medium, large-v3-turbo).
                       Si None, utilise le modèle de la config.
        """
        self.config = config
        self.model_name = model_name or config.model
        self._model_lock = threading.Lock()  # Verrou pour garantir l'usage exclusif du modèle
        
        # Lazy loading : les modèles seront chargés quand nécessaire
        self.model = None
        self.stereo_diarization_service = None
        self.pyannote_diarization_service = None
    
    def _load_model(self):
        """Charge le modèle Whisper"""
        # Construire le chemin du modèle si c'est un nom simple (tiny, base, etc.)
        model_path = self.model_name
        
        # Si c'est un nom simple (tiny, base, small, medium, large, large-v3, large-v3-turbo), construire le chemin
        if model_path in ["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"]:
            # Tous les modèles sont montés dans /app/models via le volume podman:
            # /opt/vocalyx/shared/models -> /app/models
            # Donc large-v3 et large-v3-turbo utilisent le même chemin que les autres modèles
            model_path = f"./models/transcribe/openai-whisper-{self.model_name}"
        
        # Convertir les chemins relatifs en chemins absolus
        # faster-whisper interprète les chemins relatifs comme des repo_id HuggingFace, donc
        # on ne touche qu'aux chemins relatifs qui ne sont PAS déjà des repo_id HuggingFace.
        if model_path.startswith("./"):
            # Enlever le préfixe ./ et construire le chemin absolu
            relative_path = model_path[2:]  # Enlever "./"
            # Utiliser /app comme base (WORKDIR du conteneur Docker)
            model_path = f"/app/{relative_path}"
        elif (
            not model_path.startswith("/")
            and not model_path.startswith("openai/")
        ):
            # Si c'est un chemin relatif sans ./ (ex: "models/...")
            # et que ce n'est pas un repo HuggingFace, le convertir en absolu
            model_path = f"/app/{model_path}"
        
        logger.info(f"🚀 Loading Whisper model: {model_path} (requested: {self.model_name})")
        logger.info(f"📊 Device: {self.config.device} | Compute: {self.config.compute_type}")
        
        self.model = WhisperModel(
            model_path,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads
        )
        
        logger.info(f"✅ Whisper model loaded successfully")
        best_of = getattr(self.config, 'best_of', self.config.beam_size)
        logger.info(f"⚙️ VAD: {self.config.vad_enabled} | Beam size: {self.config.beam_size} | Best of: {best_of}")
    
    def _load_diarization_services(self):
        """Charge les services de diarisation (stéréo + pyannote pour mono)"""
        # Charger le service stéréo (toujours disponible, léger)
        if self.stereo_diarization_service is None:
            try:
                logger.info("🎯 Initializing stereo diarization service (lightweight, no ML models)")
                self.stereo_diarization_service = StereoDiarizationService(self.config)
                logger.info("✅ Stereo diarization service initialized and ready")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize stereo diarization service: {e}")
                self.stereo_diarization_service = None
        
        # Charger le service pyannote pour mono (seulement si nécessaire, plus lourd)
        if self.pyannote_diarization_service is None:
            try:
                logger.info("🎯 Initializing pyannote diarization service (for mono audio, like WhisperX)")
                self.pyannote_diarization_service = PyannoteDiarizationService(self.config)
                logger.info("✅ Pyannote diarization service initialized and ready")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize pyannote diarization service: {e} (mono diarization will be skipped)")
                self.pyannote_diarization_service = None
        
    def transcribe_segment(self, file_path: Path, use_vad: bool = True, retry_without_vad: bool = True, initial_prompt: Optional[str] = None) -> Tuple[str, List[Dict], str]:
        """
        Transcrit un segment audio avec consommation progressive du générateur.
        
        Args:
            file_path: Chemin vers le segment audio
            use_vad: Utiliser la détection de voix
            retry_without_vad: Réessayer sans VAD en cas d'erreur
            initial_prompt: Prompt initial optionnel pour guider la transcription (comme WhisperX)
        """
        # Charger le modèle Whisper en lazy loading
        if self.model is None:
            logger.info(f"🔄 Loading Whisper model (lazy loading): {self.model_name}")
            self._load_model()
        
        segments_list = []
        text_full = ""
        info = None # Initialiser info
        
        logger.info(f"🎯 Starting Whisper transcription (VAD: {use_vad})...")
        
        try:
            vad_params = None
            if use_vad:
                # Utiliser les paramètres VAD de la config (plus cohérent)
                vad_params = dict(
                    threshold=self.config.vad_threshold,
                    min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                    min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                    speech_pad_ms=self.config.vad_speech_pad_ms
                )
            
            # --- MODIFICATION ---
            # Utiliser le verrou pour garantir l'usage exclusif du modèle (transcribe + consommation)
            # Cela évite les conflits entre threads Celery (concurrency=2)
            # Exécuter dans un thread séparé avec timeout global pour éviter les blocages infinis
            result_container = {'segments_list_raw': None, 'info': None, 'error': None}
            
            def transcribe_with_timeout():
                """Fonction exécutée dans un thread séparé avec timeout global"""
                try:
                    with self._model_lock:
                        logger.info(f"🔒 Acquired model lock, starting transcription...")
                        transcribe_start_time = time.time()
                        
                        # Paramètres optimisés pour CPU
                        # initial_prompt: comme WhisperX, guide la première fenêtre de transcription (~30s)
                        # Dans WhisperX, initial_prompt est encodé et ajouté aux tokens initiaux
                        # Ici, faster-whisper gère automatiquement l'encodage
                        transcribe_kwargs = {
                            "language": self.config.language or None,
                            "task": "transcribe",
                            "beam_size": self.config.beam_size,  # 1 pour CPU (greedy search)
                            "best_of": getattr(self.config, 'best_of', self.config.beam_size),  # 1 pour CPU
                            "temperature": self.config.temperature,
                            "vad_filter": use_vad,  # VAD intégré de faster-whisper (optimisé)
                            "vad_parameters": vad_params,
                            "word_timestamps": False,  # Désactivé pour CPU (plus rapide)
                            "condition_on_previous_text": False,  # Désactivé pour CPU (plus rapide)
                        }
                        
                        # Ajouter initial_prompt si fourni (comme WhisperX)
                        if initial_prompt is not None:
                            transcribe_kwargs["initial_prompt"] = initial_prompt
                            logger.info(f"📝 Using initial_prompt: {initial_prompt[:50]}..." if len(initial_prompt) > 50 else f"📝 Using initial_prompt: {initial_prompt}")
                        
                        segments, info = self.model.transcribe(
                            str(file_path),
                            **transcribe_kwargs
                        )
                        
                        transcribe_time = time.time() - transcribe_start_time
                        logger.info(f"🎯 Whisper inference completed in {transcribe_time:.1f}s, consuming generator...")
                        
                        # Consommer le générateur progressivement
                        start_consume_time = time.time()
                        logger.info(f"🔄 Starting to consume generator (this may take a moment)...")
                        
                        segments_list_raw = []
                        segment_count = 0
                        last_log_time = time.time()
                        
                        for seg in segments:
                            segments_list_raw.append(seg)
                            segment_count += 1
                            
                            # Logger tous les 10 segments OU toutes les 2 secondes pour suivre la progression
                            current_time = time.time()
                            elapsed = current_time - start_consume_time
                            if segment_count % 10 == 0 or (current_time - last_log_time) >= 2.0:
                                logger.info(f"🔄 Consumed {segment_count} segments so far (elapsed: {elapsed:.1f}s)...")
                                last_log_time = current_time
                        
                        consume_time = time.time() - start_consume_time
                        total_time = time.time() - transcribe_start_time
                        logger.info(f"✅ Generator consumed, got {len(segments_list_raw)} segments in {consume_time:.1f}s (total: {total_time:.1f}s)")
                        logger.info(f"🔓 Releasing model lock")
                        
                        result_container['segments_list_raw'] = segments_list_raw
                        result_container['info'] = info
                except Exception as e:
                    logger.error(f"❌ Error in transcription thread: {e}", exc_info=True)
                    result_container['error'] = e
            
            # Exécuter dans un thread avec timeout configurable
            transcription_timeout = getattr(self.config, 'transcription_timeout', 300)  # Défaut: 5 minutes
            timeout_minutes = transcription_timeout / 60
            logger.info(f"🚀 Starting transcription in separate thread with {timeout_minutes:.1f}min timeout ({transcription_timeout}s)...")
            thread = threading.Thread(target=transcribe_with_timeout, daemon=True)
            thread.start()
            thread.join(timeout=transcription_timeout)
            
            if thread.is_alive():
                # Le thread est toujours en vie après le timeout = blocage
                logger.error(f"❌ TIMEOUT: Transcription thread still alive after {transcription_timeout}s - forcing error")
                raise TimeoutError(f"Transcription timeout after {transcription_timeout}s - generator appears to be blocked")
            
            # Vérifier les résultats
            if result_container['error']:
                raise result_container['error']
            
            if result_container['segments_list_raw'] is None:
                logger.error(f"❌ Transcription thread completed but no segments returned")
                raise RuntimeError("Transcription completed but no segments were returned")
            
            segments_list_raw = result_container['segments_list_raw']
            info = result_container['info']
            # --- FIN MODIFICATION ---
            
        except Exception as e:
            # Gérer les erreurs de transcription ou de consommation
            logger.error(f"❌ Error during transcription/consumption: {e}", exc_info=True)
            
            if use_vad and retry_without_vad:
                logger.warning(f"⚠️ Retrying WITHOUT VAD...")
                return self.transcribe_segment(file_path, use_vad=False, retry_without_vad=False, initial_prompt=initial_prompt)
            else:
                raise Exception(f"Transcription failed: {e}")
        
        # S'assurer que 'info' a été défini
        if info is None:
            logger.error("❌ 'info' n'a pas été retourné par model.transcribe(), impossible de détecter la langue.")
            raise Exception("Transcription failed: 'info' object is None")

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
    
    def _transcribe_sequential(
        self, 
        segment_paths: List[Path], 
        use_vad: bool, 
        log_prefix: str,
        initial_prompt: Optional[str] = None
    ) -> Tuple[List[Dict], str, Optional[str]]:
        """
        Transcrit plusieurs segments de manière séquentielle dans le processus Celery.
        
        Args:
            segment_paths: Liste des chemins vers les segments audio
            use_vad: Utiliser la détection de voix
            log_prefix: Préfixe pour les logs
            initial_prompt: Prompt initial optionnel (utilisé uniquement pour le premier segment, comme WhisperX)
            
        Returns:
            Tuple contenant:
            - full_segments: Liste de tous les segments avec timestamps ajustés
            - full_text: Texte complet concaténé
            - language_detected: Langue détectée
        """
        full_segments = []
        full_text = ""
        language_detected = None
        num_segments = len(segment_paths)
        
        logger.info(f"{log_prefix}⚡ Starting sequential transcription: {num_segments} segments")
        
        # Transcription séquentielle directement dans le processus Celery
        # Comme WhisperX, initial_prompt n'est utilisé que pour le premier segment
        # (il guide uniquement la première fenêtre de transcription, pas les suivantes)
        time_offset = 0.0
        for i, segment_path in enumerate(segment_paths):
            try:
                # Utiliser initial_prompt uniquement pour le premier segment (comme WhisperX)
                segment_prompt = initial_prompt if i == 0 else None
                text, segments_list, lang = self.transcribe_segment(segment_path, use_vad, initial_prompt=segment_prompt)
                
                # Ajuster les timestamps avec l'offset
                for seg in segments_list:
                    seg["start"] = round(seg["start"] + time_offset, 2)
                    seg["end"] = round(seg["end"] + time_offset, 2)
                    full_segments.append(seg)
                
                # Mettre à jour l'offset pour le prochain segment
                if segments_list:
                    time_offset = full_segments[-1]["end"]
                
                full_text += text + " "
                
                if language_detected is None:
                    language_detected = lang
                
                logger.info(f"{log_prefix}✅ Segment {i+1}/{num_segments} completed")
                    
            except Exception as e:
                logger.error(f"{log_prefix}❌ Error transcribing segment {i+1}: {e}", exc_info=True)
                # Continuer avec les autres segments en cas d'erreur
                continue
        
        logger.info(f"{log_prefix}✅ Sequential transcription completed: {len(full_segments)} total segments")
        
        return full_segments, full_text.strip(), language_detected
    
    def transcribe(self, file_path: str, use_vad: bool = True, use_diarization: bool = False, transcription_id: str = None, initial_prompt: Optional[str] = None) -> Dict:
        """
        Transcrit un fichier audio (point d'entrée principal).
        
        Args:
            file_path: Chemin vers le fichier audio
            use_vad: Utiliser la détection de voix
            use_diarization: Activer la diarisation des locuteurs
            transcription_id: ID de la transcription (pour les logs)
            initial_prompt: Prompt initial optionnel pour guider la transcription (comme WhisperX)
            
        Returns:
            dict: Résultats de la transcription
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Format des logs avec transcription_id si disponible
        log_prefix = f"[{transcription_id}] " if transcription_id else ""
        
        logger.info(f"{log_prefix}📁 Processing file: {file_path.name} | VAD requested: {use_vad}")
        
        # Charger le modèle Whisper en lazy loading
        if self.model is None:
            logger.info(f"{log_prefix}🔄 Loading Whisper model (lazy loading): {self.model_name}")
            self._load_model()
        
        # Charger les services de diarisation seulement si nécessaire
        if use_diarization:
            if self.stereo_diarization_service is None or self.pyannote_diarization_service is None:
                logger.info(f"{log_prefix}🔄 Loading diarization services (lazy loading)...")
                self._load_diarization_services()
        
        segment_paths = []
        processed_path_mono = None
        processed_path_stereo = None
        
        try:
            start_time = time.time()
            
            # 1. Obtenir la durée réelle de l'audio
            original_duration = get_audio_duration(file_path)
            logger.info(f"{log_prefix}📏 Audio duration: {original_duration}s")
            
            # 2. Pré-traitement audio (normalisation, conversion mono 16kHz, préservation stéréo)
            preprocessed = preprocess_audio(file_path, preserve_stereo_for_diarization=use_diarization)
            processed_path_mono = preprocessed['mono']
            processed_path_stereo = preprocessed.get('stereo')
            is_stereo = preprocessed.get('is_stereo', False)
            
            if is_stereo and processed_path_stereo:
                logger.info(f"{log_prefix}✨ Audio preprocessed: MONO for Whisper, STEREO preserved for diarization")
            else:
                logger.info(f"{log_prefix}✨ Audio preprocessed: MONO (stereo not detected or diarization disabled)")
            
            # 3. Découpe intelligente (si nécessaire) - utilise la version mono pour Whisper avec taille adaptative
            segment_paths = split_audio_intelligent(
                processed_path_mono,
                use_vad=use_vad,
                segment_length_ms=self.config.segment_length_ms  # Taille adaptative selon CPU
            )
            logger.info(f"{log_prefix}🔪 Created {len(segment_paths)} segment(s) (adaptive size: {self.config.segment_length_ms}ms)")
            
            # 4. Transcription (séquentielle dans le processus Celery)
            full_text = ""
            full_segments = []
            language_detected = None
            
            # Transcription séquentielle pour plusieurs segments
            if len(segment_paths) > 1:
                logger.info(f"{log_prefix}⚡ Sequential transcription: {len(segment_paths)} segments")
                full_segments, full_text, language_detected = self._transcribe_sequential(
                    segment_paths, use_vad, log_prefix, initial_prompt
                )
            else:
                # Transcription séquentielle pour un seul segment (plus simple)
                logger.info(f"{log_prefix}🎤 Transcribing single segment...")
                text, segments_list, lang = self.transcribe_segment(segment_paths[0], use_vad=use_vad, initial_prompt=initial_prompt)
                
                full_segments = segments_list
                full_text = text
                language_detected = lang
            
            # 5. Diarisation (stéréo ou mono selon le format audio)
            if use_diarization:
                logger.info(f"{log_prefix}🎤 Running speaker diarization...")
                try:
                    # Choisir le service selon le format audio
                    if is_stereo and processed_path_stereo:
                        # Audio stéréo : utiliser la diarisation stéréo (légère et rapide)
                        if self.stereo_diarization_service:
                            logger.info(f"{log_prefix}🎯 Using STEREO diarization (lightweight: one channel per speaker)")
                            diarization_audio_path = processed_path_stereo
                            diarization_segments = self.stereo_diarization_service.diarize(diarization_audio_path)
                            
                            if diarization_segments:
                                # Assigner les locuteurs aux segments de transcription
                                full_segments = self.stereo_diarization_service.assign_speakers_to_segments(
                                    full_segments,
                                    diarization_segments
                                )
                                logger.info(f"{log_prefix}✅ Stereo diarization completed and assigned to segments")
                            else:
                                logger.warning(f"{log_prefix}⚠️ Stereo diarization returned no segments")
                        else:
                            logger.warning(f"{log_prefix}⚠️ Stereo diarization requested but service not available")
                    else:
                        # Audio mono : utiliser la diarisation pyannote (comme WhisperX)
                        if self.pyannote_diarization_service:
                            logger.info(f"{log_prefix}🎯 Using PYANNOTE diarization for MONO audio (like WhisperX)")
                            diarization_audio_path = processed_path_mono
                            diarization_segments = self.pyannote_diarization_service.diarize(diarization_audio_path)
                            
                            if diarization_segments:
                                # Assigner les locuteurs aux segments de transcription
                                full_segments = self.pyannote_diarization_service.assign_speakers_to_segments(
                                    full_segments,
                                    diarization_segments
                                )
                                logger.info(f"{log_prefix}✅ Pyannote diarization completed and assigned to segments")
                            else:
                                logger.warning(f"{log_prefix}⚠️ Pyannote diarization returned no segments")
                        else:
                            logger.warning(f"{log_prefix}⚠️ Pyannote diarization requested but service not available (mono audio)")
                except Exception as e:
                    logger.error(f"{log_prefix}❌ Error during diarization: {e}", exc_info=True)
                    # Continuer sans diarisation en cas d'erreur
            else:
                logger.info(f"{log_prefix}ℹ️ Diarization disabled for this transcription")
            
            processing_time = round(time.time() - start_time, 2)
            speed_ratio = round(original_duration / processing_time, 2) if processing_time > 0 else 0
            
            logger.info(
                f"{log_prefix}✅ Transcription completed | "
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
                # Nettoyer la version mono
                if 'processed_path_mono' in locals() and processed_path_mono and processed_path_mono != file_path:
                    if processed_path_mono.exists():
                        processed_path_mono.unlink()
                
                # Nettoyer la version stéréo si créée
                if 'processed_path_stereo' in locals() and processed_path_stereo:
                    if processed_path_stereo.exists():
                        processed_path_stereo.unlink()
                
                # Nettoyer les segments
                for seg_path in segment_paths:
                    if seg_path.exists():
                        seg_path.unlink()
                
                log_prefix = f"[{transcription_id}] " if transcription_id else ""
                logger.debug(f"{log_prefix}🧹 Temporary files cleaned")
            except Exception as e:
                log_prefix = f"[{transcription_id}] " if transcription_id else ""
                logger.warning(f"{log_prefix}⚠️ Cleanup error: {e}")
