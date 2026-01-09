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
        self.diarization_service = None
    
    def _load_model(self):
        """Charge le modèle Whisper"""
        # Construire le chemin du modèle si c'est un nom simple (tiny, base, etc.)
        model_path = self.model_name
        
        # Si c'est un nom simple (tiny, base, small, medium, large-v3-turbo), construire le chemin
        if model_path in ["tiny", "base", "small", "medium", "large", "large-v3-turbo"]:
            # Pour large-v3-turbo, utiliser le nom complet
            if model_path == "large-v3-turbo":
                model_path = f"./models/transcribe/openai-whisper-large-v3-turbo"
            else:
                model_path = f"./models/transcribe/openai-whisper-{self.model_name}"
        
        # Convertir les chemins relatifs en chemins absolus
        # faster-whisper interprète les chemins relatifs comme des repo_id HuggingFace
        if model_path.startswith("./"):
            # Enlever le préfixe ./ et construire le chemin absolu
            relative_path = model_path[2:]  # Enlever "./"
            # Utiliser /app comme base (WORKDIR du conteneur Docker)
            model_path = f"/app/{relative_path}"
        elif not model_path.startswith("/") and not model_path.startswith("openai/"):
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
    
    def _load_diarization_service(self):
        """Charge le service de diarisation stéréo (seulement si nécessaire)"""
        if self.diarization_service is not None:
            return  # Déjà chargé
        
        try:
            # Utiliser uniquement la diarisation stéréo (légère et rapide, sans modèle ML)
            logger.info("🎯 Using stereo diarization (lightweight, no ML models)")
            self.diarization_service = StereoDiarizationService(self.config)
            logger.info("✅ Stereo diarization service initialized and ready")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize diarization service: {e} (will be skipped if requested)")
            self.diarization_service = None
        
    def transcribe_segment(self, file_path: Path, use_vad: bool = True, retry_without_vad: bool = True, initial_prompt: Optional[str] = None) -> Tuple[str, List[Dict], str]:
        """
        Transcrit un segment audio avec consommation progressive du générateur.
        
        Args:
            file_path: Chemin vers le fichier audio
            use_vad: Utiliser la détection de voix
            retry_without_vad: Réessayer sans VAD en cas d'erreur
            initial_prompt: Prompt de contexte pour guider Whisper
        """
        # Charger le modèle Whisper en lazy loading
        if self.model is None:
            logger.info(f"🔄 Loading Whisper model (lazy loading): {self.model_name}")
            self._load_model()
        
        segments_list = []
        text_full = ""
        info = None # Initialiser info
        
        # ⚠️ IMPORTANT : Respecter le paramètre use_vad passé en argument
        # Si use_vad=False a été passé (par exemple avec initial_prompt), ne pas réactiver le VAD
        vad_params = None
        use_vad_effective = use_vad
        
        if use_vad:
            # Utiliser les paramètres VAD de la config (plus cohérent)
            vad_params = dict(
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                speech_pad_ms=self.config.vad_speech_pad_ms
            )
        # Si use_vad=False, ne pas activer le VAD même avec initial_prompt
        # Le paramètre use_vad a déjà été géré par l'appelant (transcribe())
        
        logger.info(f"🎯 Starting Whisper transcription (VAD: {use_vad_effective}, initial_prompt: {initial_prompt if initial_prompt else '(none)'})...")
        
        try:
            
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
                        # Log pour vérifier que le VAD est bien désactivé avec initial_prompt
                        if initial_prompt:
                            logger.info(f"🔍 DEBUG: Transcribing with initial_prompt | vad_filter={use_vad_effective} | vad_params={vad_params}")
                        
                        segments, info = self.model.transcribe(
                            str(file_path),
                            language=self.config.language or None,
                            task="transcribe",
                            beam_size=self.config.beam_size,  # 1 pour CPU (greedy search)
                            best_of=getattr(self.config, 'best_of', self.config.beam_size),  # 1 pour CPU
                            temperature=self.config.temperature,
                            vad_filter=use_vad_effective,  # VAD désactivé si initial_prompt
                            vad_parameters=vad_params,
                            word_timestamps=False,  # Désactivé pour CPU (plus rapide)
                            condition_on_previous_text=False,  # Désactivé pour CPU (plus rapide)
                            initial_prompt=initial_prompt  # Prompt de contexte pour guider la transcription
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
        time_offset = 0.0
        for i, segment_path in enumerate(segment_paths):
            try:
                # ⚠️ IMPORTANT : Appliquer initial_prompt uniquement au premier segment (index 0)
                # Le prompt initial doit guider le contexte du début, pas de chaque segment
                use_prompt_for_this_segment = initial_prompt if (i == 0 and initial_prompt) else None
                text, segments_list, lang = self.transcribe_segment(segment_path, use_vad, initial_prompt=use_prompt_for_this_segment)
                
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
            initial_prompt: Prompt de contexte pour guider Whisper
            
        Returns:
            dict: Résultats de la transcription
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Format des logs avec transcription_id si disponible
        log_prefix = f"[{transcription_id}] " if transcription_id else ""
        
        logger.info(f"{log_prefix}📁 Processing file: {file_path.name} | VAD: {use_vad} | Initial prompt: {initial_prompt if initial_prompt else '(none)'}")
        
        # Charger le modèle Whisper en lazy loading
        if self.model is None:
            logger.info(f"{log_prefix}🔄 Loading Whisper model (lazy loading): {self.model_name}")
            self._load_model()
        
        # Charger le service de diarisation seulement si nécessaire
        if use_diarization and self.diarization_service is None:
            logger.info(f"{log_prefix}🔄 Loading diarization service (lazy loading)...")
            self._load_diarization_service()
        
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
            # ⚠️ IMPORTANT : Si un initial_prompt est fourni, NE PAS segmenter l'audio
            # Le prompt initial doit être appliqué à l'audio complet, pas à des segments individuels
            if initial_prompt:
                logger.info(f"{log_prefix}🔍 Initial prompt provided → Skipping segmentation (will transcribe full audio as single segment)")
                segment_paths = [processed_path_mono]  # Traiter l'audio complet comme un seul segment
            else:
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
                # Si initial_prompt est fourni, il ne devrait pas y avoir plusieurs segments (segmentation désactivée)
                # Mais par sécurité, on ne passe le prompt qu'au premier segment si on a plusieurs segments
                full_segments, full_text, language_detected = self._transcribe_sequential(
                    segment_paths, use_vad, log_prefix, initial_prompt=None  # Pas de prompt pour segments multiples
                )
            else:
                # Transcription séquentielle pour un seul segment (plus simple)
                logger.info(f"{log_prefix}🎤 Transcribing single segment...")
                # ⚠️ IMPORTANT : Si initial_prompt est fourni, désactiver le VAD pour garantir la transcription complète
                # Le VAD peut être trop agressif et filtrer le début, même avec un prompt
                use_vad_for_transcription = use_vad if not initial_prompt else False
                if initial_prompt and not use_vad_for_transcription:
                    logger.info(f"{log_prefix}🔍 VAD disabled for transcription with initial_prompt to ensure complete transcription from start")
                text, segments_list, lang = self.transcribe_segment(segment_paths[0], use_vad=use_vad_for_transcription, initial_prompt=initial_prompt)
                
                full_segments = segments_list
                full_text = text
                language_detected = lang
            
            # 5. Diarisation stéréo (si activée pour cette transcription)
            if use_diarization:
                if self.diarization_service:
                    logger.info(f"{log_prefix}🎤 Running speaker diarization...")
                    try:
                        # Utiliser la version stéréo pour la diarisation si disponible (optimal pour séparation des locuteurs)
                        # Sinon utiliser la version mono
                        diarization_audio_path = processed_path_stereo if processed_path_stereo else processed_path_mono
                        if processed_path_stereo:
                            logger.info(f"{log_prefix}🎯 Using STEREO audio for diarization (optimal: one channel per speaker)")
                        else:
                            logger.info(f"{log_prefix}🎯 Using MONO audio for diarization")
                        
                        diarization_segments = self.diarization_service.diarize(diarization_audio_path)
                        
                        if diarization_segments:
                            # Assigner les locuteurs aux segments de transcription
                            full_segments = self.diarization_service.assign_speakers_to_segments(
                                full_segments,
                                diarization_segments
                            )
                            logger.info(f"{log_prefix}✅ Speaker diarization completed and assigned to segments")
                        else:
                            logger.warning(f"{log_prefix}⚠️ Diarization returned no segments")
                    except Exception as e:
                        logger.error(f"{log_prefix}❌ Error during diarization: {e}", exc_info=True)
                        # Continuer sans diarisation en cas d'erreur
                else:
                    logger.warning(f"{log_prefix}⚠️ Diarization requested but service not available")
            
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
