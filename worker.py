"""
vocalyx-transcribe/worker.py
Worker Celery pour la transcription audio
"""

import logging
import time
import os
import psutil
import json
import redis
from pathlib import Path
from datetime import datetime
from celery.signals import worker_init
from celery.worker.control import Panel
from celery import Celery

from config import Config
from infrastructure.api.api_client import VocalyxAPIClient
from infrastructure.redis.redis_manager import RedisCompressionManager, RedisTranscriptionManager
from infrastructure.models.model_cache import ModelCache
from application.services.diarization_merger import DiarizationMerger
from audio_utils import split_audio_intelligent, preprocess_audio, get_audio_duration
from pydub import AudioSegment

config = Config()

from logging_config import setup_logging, setup_colored_logging

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

# Variables globales pour les services (singletons par worker)
_api_client = None
_redis_client = None
_redis_manager = None
_model_cache = None

# Variables globales pour psutil
WORKER_PROCESS = None
WORKER_START_TIME = None


@worker_init.connect
def on_worker_init(**kwargs):
    """Initialise psutil quand le worker démarre."""
    global WORKER_PROCESS, WORKER_START_TIME
    try:
        WORKER_PROCESS = psutil.Process(os.getpid())
        WORKER_START_TIME = datetime.now()
        WORKER_PROCESS.cpu_percent(interval=None)
        logger.info(f"Worker {WORKER_PROCESS.pid} initialisé pour monitoring psutil.")
        # Loguer la configuration Redis importante pour les transcriptions distribuées
        try:
            redis_ttl = getattr(config, 'redis_transcription_ttl', None)
            logger.info(f"🔧 Redis transcription TTL configuré: {redis_ttl}s")
        except Exception as cfg_err:
            logger.warning(f"⚠️ Impossible de lire redis_transcription_ttl: {cfg_err}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de psutil: {e}")


def get_redis_client():
    """Obtient un client Redis pour stocker les résultats des segments"""
    global _redis_client
    if _redis_client is None:
        redis_url = getattr(config, 'redis_transcription_url', None)
        if not redis_url:
            base_url = config.celery_broker_url.rsplit('/', 1)[0]
            redis_url = f"{base_url}/2"
        
        logger.info(f"🔌 Initializing Redis transcription client: {redis_url}")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        
        try:
            _redis_client.ping()
            logger.info(f"✅ Redis transcription client connected successfully: {redis_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis transcription: {redis_url} - {e}")
            raise
    
    return _redis_client


def get_redis_manager() -> RedisTranscriptionManager:
    """Obtient le gestionnaire Redis pour les opérations de transcription"""
    global _redis_manager
    if _redis_manager is None:
        redis_client = get_redis_client()
        compression = RedisCompressionManager(
            enabled=getattr(config, 'redis_transcription_compress', True)
        )
        _redis_manager = RedisTranscriptionManager(redis_client, compression)
    return _redis_manager


def get_api_client():
    """Charge le client API (une fois par worker)"""
    global _api_client
    if _api_client is None:
        logger.info(f"Initialisation du client API pour ce worker ({config.instance_name})...")
        _api_client = VocalyxAPIClient(config)
    return _api_client


def get_transcription_service(model_name: str = 'small'):
    """
    Charge le service de transcription avec cache par modèle.
    
    Args:
        model_name: Nom du modèle Whisper (tiny, base, small, medium, large-v3-turbo) ou chemin
        
    Returns:
        TranscriptionService: Service de transcription avec le modèle demandé
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache(max_models=10)
    
    return _model_cache.get(model_name, config)


def trigger_enrichment_task(transcription_id: str, api_client: VocalyxAPIClient):
    """Déclenche la tâche d'enrichissement si demandée"""
    try:
        from celery import current_app as celery_current_app
        logger.info(f"[{transcription_id}] 🤖 Enrichment requested, triggering enrichment task...")
        enrich_task = celery_current_app.send_task(
            'enrich_transcription',
            args=[transcription_id],
            queue='enrichment',
            countdown=1
        )
        logger.info(f"[{transcription_id}] ✅ Enrichment task enqueued: {enrich_task.id}")
    except Exception as enrich_error:
        logger.warning(f"[{transcription_id}] ⚠️ Failed to enqueue enrichment task: {enrich_error}")
        try:
            api_client.update_transcription(transcription_id, {
                "enrichment_status": "error",
                "enrichment_error": f"Failed to enqueue enrichment: {str(enrich_error)}"
            })
        except:
            pass


# Créer l'application Celery (connexion au même broker que vocalyx-api)
celery_app = Celery(
    'vocalyx-transcribe',
    broker=config.celery_broker_url,
    backend=config.celery_result_backend
)

# ... (votre configuration Celery conf.update reste inchangée) ...
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Configuration du format de logging pour Celery
    worker_log_format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    worker_task_log_format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    worker_log_datefmt='%Y-%m-%d %H:%M:%S',
    # Désactiver le warning de sécurité pour les containers Docker (on tourne en root par design)
    worker_disable_rate_limits=False,
    # Ignorer le warning de sécurité root dans les containers Docker
    worker_hijack_root_logger=False,
)


@Panel.register(
    name='get_worker_health',
    alias='health'
)
def get_worker_health_handler(state, **kwargs):
    """
    Handler pour la commande de contrôle 'get_worker_health'.
    Ne retourne que les stats psutil (CPU/RAM/Uptime).
    """
    if WORKER_PROCESS is None:
        logger.warning("get_worker_health_handler appelé avant initialisation de psutil.")
        return {'error': 'Worker not initialized'}
    
    try:
        mem_info = WORKER_PROCESS.memory_info()
        uptime_seconds = (datetime.now() - WORKER_START_TIME).total_seconds()
        
        # Les stats métier (audio traité) sont calculées par l'API
        
        health_data = {
            'pid': WORKER_PROCESS.pid,
            'cpu_percent': WORKER_PROCESS.cpu_percent(interval=None),
            'memory_rss_bytes': mem_info.rss,
            'memory_percent': WORKER_PROCESS.memory_percent(),
            'uptime_seconds': uptime_seconds
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Erreur dans get_worker_health_handler: {e}", exc_info=True)
        return {'error': str(e)}

@celery_app.task(
    bind=True,
    name='transcribe_audio',  # Même nom que dans l'API pour compatibilité
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=1800,
    time_limit=2100,
    acks_late=True,
    reject_on_worker_lost=True
)
def transcribe_audio_task(self, transcription_id: str, use_distributed: bool = None):
    """
    Tâche de transcription exécutée par le worker.
    
    Si use_distributed=True ou si l'audio dépasse le seuil, cette tâche va
    déléguer à orchestrate_distributed_transcription au lieu de traiter directement.
    """
    
    # Assure que le client API est initialisé
    api_client = get_api_client()
    
    # 1. Récupérer les informations de la transcription depuis l'API
    logger.info(f"[{transcription_id}] 📡 Fetching transcription data from API...")
    transcription = api_client.get_transcription(transcription_id)
    
    if not transcription:
        raise ValueError(f"Transcription {transcription_id} not found")
    
    file_path = transcription.get('file_path')
    
    # Vérifier si un initial_prompt est fourni
    initial_prompt = transcription.get('initial_prompt')
    # Normaliser le prompt : convertir les chaînes vides en None
    if initial_prompt is not None and isinstance(initial_prompt, str) and initial_prompt.strip() == "":
        initial_prompt = None
    has_initial_prompt = initial_prompt is not None and initial_prompt.strip() != ""
    
    # Vérifier si on doit utiliser le mode distribué
    if use_distributed is None:
        try:
            # ⚠️ MODIFICATION : Permettre le mode distribué même avec initial_prompt
            # En mode distribué, l'initial_prompt est utilisé uniquement pour le premier segment (segment_index == 0)
            # Cela fonctionne mieux que le mode non-distribué qui ignore souvent le début de l'audio
            if file_path:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    import soundfile as sf
                    duration = sf.info(str(file_path_obj)).duration
                    # Seuil configurable pour activer le mode distribué
                    # 0 = désactiver le mode distribué (traitement classique uniquement)
                    min_duration = config.distributed_min_duration_seconds
                    use_distributed = min_duration > 0 and duration > min_duration
                    logger.info(
                        f"[{transcription_id}] 📊 DISTRIBUTION DECISION (worker) | "
                        f"Duration: {duration:.1f}s | "
                        f"Threshold: {min_duration}s {'(distribué désactivé)' if min_duration == 0 else ''} | "
                        f"Mode: {'DISTRIBUTED' if use_distributed else 'CLASSIC'}"
                    )
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Could not determine distribution mode: {e}")
            use_distributed = False
    
    # Si mode distribué, déléguer à orchestrate_distributed_transcription
    if use_distributed and file_path:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            logger.info(f"[{transcription_id}] 🚀 DISTRIBUTED MODE | Delegating to orchestrate_distributed_transcription | Worker: {config.instance_name}")
            
            from celery import current_app as celery_current_app
            orchestrate_task = celery_current_app.send_task(
                'orchestrate_distributed_transcription',
                args=[transcription_id, str(file_path)],
                queue='transcription',
                countdown=1
            )
            
            logger.info(f"[{transcription_id}] ✅ DISTRIBUTED MODE | Orchestration task enqueued: {orchestrate_task.id}")
            
            return {
                "transcription_id": transcription_id,
                "task_id": self.request.id,
                "orchestration_task_id": orchestrate_task.id,
                "status": "queued_distributed",
                "mode": "distributed"
            }
    
    # MODE CLASSIQUE : Déléguer à la fonction dédiée
    try:
        return self._transcribe_classic_mode(
            transcription_id=transcription_id,
            transcription=transcription,
            file_path=file_path,
            initial_prompt=initial_prompt,
            api_client=api_client
        )
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Error in transcribe_audio_task: {e}", exc_info=True)
        # Mettre à jour le statut à "error"
        try:
            api_client_on_error = get_api_client()
            api_client_on_error.update_transcription(transcription_id, {
                "status": "error",
                "error_message": str(e)
            })
        except Exception as update_error:
            logger.error(f"[{transcription_id}] Failed to update error status: {update_error}")
        
        # Retry si possible
        if self.request.retries < self.max_retries:
            logger.warning(f"[{transcription_id}] ⏳ Retrying in {self.default_retry_delay}s...")
            raise self.retry(exc=e)
        
        # Si toutes les tentatives échouent
        logger.error(f"[{transcription_id}] ⛔ All retries exhausted")
        
        # Supprimer le fichier .wav original même en cas d'échec
        if file_path:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists() and file_path_obj.suffix.lower() == '.wav':
                    file_path_obj.unlink()
                    logger.info(f"[{transcription_id}] 🗑️ Original audio file deleted after error: {file_path}")
                elif file_path_obj.exists():
                    logger.debug(f"[{transcription_id}] ℹ️ File not deleted (not .wav): {file_path}")
            except Exception as delete_error:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to delete original audio file after error: {delete_error}")
        
        return {
            "status": "error",
            "transcription_id": transcription_id,
            "error": str(e)
        }

def _transcribe_classic_mode(self, transcription_id: str, transcription: dict, file_path: str, initial_prompt, api_client):
    """
    Traite la transcription en mode classique (non-distribué).
    Un seul worker traite l'audio complet de manière séquentielle.
    
    Args:
        transcription_id: ID de la transcription
        transcription: Données de la transcription depuis l'API
        file_path: Chemin vers le fichier audio
        initial_prompt: Prompt initial pour guider Whisper (optionnel)
        api_client: Client API pour mettre à jour la transcription
    """
    logger.info(
        f"[{transcription_id}] 🎯 CLASSIC MODE STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"Mode: Single worker processing entire audio (non-distributed)"
    )
    
    # ⚠️ IMPORTANT : Enregistrer le temps de début RÉEL du traitement (pas de la création de la tâche)
    # Ce temps exclut le temps d'attente dans la file Celery
    processing_start_time = time.time()
    
    try:
        use_vad = transcription.get('vad_enabled', True)
        use_diarization = transcription.get('diarization_enabled', False)
        whisper_model = transcription.get('whisper_model', 'small')  # Récupérer le modèle choisi
        
        logger.info(f"[{transcription_id}] 📁 File: {file_path} | VAD: {use_vad} | Diarization: {use_diarization} | Model: {whisper_model}")
        
        # 2. Mettre à jour le statut à "processing" (début réel du traitement)
        # ⚠️ IMPORTANT : Le statut "processing" indique que le worker a commencé le traitement réel
        # Le temps de traitement sera calculé à partir de maintenant
        # ✅ NOUVEAU : Enregistrer le temps de début réel du traitement
        # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
        processing_start_datetime = datetime.utcnow()
        
        api_client.update_transcription(transcription_id, {
            "status": "processing",
            "worker_id": config.instance_name,
            "processing_start_time": processing_start_datetime.isoformat()  # ✅ NOUVEAU
        })
        logger.info(f"[{transcription_id}] ⚙️ Status updated to 'processing' (real processing started)")
        
        # 3. Obtenir le service de transcription avec cache de modèles (Phase 3 - Optimisation)
        # Le cache réutilise les modèles déjà chargés, évitant 5-15s de chargement
        logger.info(f"[{transcription_id}] 🎤 Getting transcription service with model: {whisper_model} (cached)")
        transcription_service = get_transcription_service(model_name=whisper_model)
        
        # 4. Exécuter la transcription
        logger.info(f"[{transcription_id}] 🎤 Starting transcription with Whisper...")
        
        # initial_prompt a déjà été récupéré et normalisé plus tôt
        logger.info(f"[{transcription_id}] 🔍 Initial prompt: {initial_prompt if initial_prompt else '(none)'}")
        
        result = transcription_service.transcribe(
            file_path=file_path,
            use_vad=use_vad,
            use_diarization=use_diarization,
            transcription_id=transcription_id,
            initial_prompt=initial_prompt
        )
        
        logger.info(f"[{transcription_id}] ✅ Transcription service completed")
        
        # ⚠️ IMPORTANT : Calculer uniquement le temps de traitement RÉEL (sans temps d'attente)
        processing_end_time = time.time()
        processing_time = round(processing_end_time - processing_start_time, 2)
        
        logger.info(
            f"[{transcription_id}] ✅ Transcription completed | "
            f"Duration: {result['duration']}s | "
            f"Processing: {processing_time}s | "
            f"Segments: {len(result['segments'])}"
        )
        
        # 4. Mettre à jour avec les résultats
        logger.info(f"[{transcription_id}] 💾 Saving results to API...")
        enrichment_requested = transcription.get('enrichment_requested', False)
        status = "transcribed" if enrichment_requested else "done"
        
        # ✅ NOUVEAU : Calculer les métriques de performance
        queued_at = transcription.get('queued_at')
        processing_start_time_str = transcription.get('processing_start_time')
        
        # Calculer le temps d'attente dans la file
        queue_wait_time = None
        if queued_at and processing_start_time_str:
            try:
                # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
                queued_dt = datetime.fromisoformat(queued_at.replace('Z', '+00:00'))
                start_dt = datetime.fromisoformat(processing_start_time_str.replace('Z', '+00:00'))
                queue_wait_time = round((start_dt - queued_dt).total_seconds(), 2)
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to calculate queue_wait_time: {e}")
        
        # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
        processing_end_datetime = datetime.utcnow()
        
        api_client.update_transcription(transcription_id, {
            "status": status,
            "text": result["text"],
            "segments": json.dumps(result["segments"]),
            "language": result["language"],
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"]),
            "queue_wait_time": queue_wait_time,  # ✅ NOUVEAU
            "processing_end_time": processing_end_datetime.isoformat()  # ✅ NOUVEAU
        })
        
        logger.info(f"[{transcription_id}] 💾 Results saved to API (status: {status})")
        
        # 5. Supprimer le fichier .wav original après transcription réussie
        if file_path:
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists() and file_path_obj.suffix.lower() == '.wav':
                    file_path_obj.unlink()
                    logger.info(f"[{transcription_id}] 🗑️ Original audio file deleted: {file_path}")
                elif file_path_obj.exists():
                    logger.debug(f"[{transcription_id}] ℹ️ File not deleted (not .wav): {file_path}")
            except Exception as delete_error:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to delete original audio file: {delete_error}")
        
        # 6. Si l'enrichissement est demandé, déclencher la tâche d'enrichissement
        if enrichment_requested:
            trigger_enrichment_task(transcription_id, api_client)
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"]),
            "enrichment_triggered": enrichment_requested
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Error in classic mode: {e}", exc_info=True)
        raise


@celery_app.task(
    bind=True,
    name='orchestrate_distributed_transcription',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True
)
def orchestrate_distributed_transcription_task(self, transcription_id: str, file_path: str):
    """
    Orchestre la transcription distribuée : découpe l'audio et crée les tâches de segments.
    
    Args:
        transcription_id: ID de la transcription
        file_path: Chemin vers le fichier audio
    """
    logger.info(
        f"[{transcription_id}] 🎼 DISTRIBUTED ORCHESTRATION STARTED | "
        f"Worker: {config.instance_name} | "
        f"File: {Path(file_path).name} | "
        f"Task ID: {self.request.id}"
    )
    
    try:
        api_client = get_api_client()
        transcription = api_client.get_transcription(transcription_id)
        
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        use_vad = transcription.get('vad_enabled', True)
        whisper_model = transcription.get('whisper_model', 'small')
        initial_prompt = transcription.get('initial_prompt')
        
        # Debug: vérifier la valeur récupérée depuis l'API
        logger.debug(
            f"[{transcription_id}] 🔍 DEBUG | Transcription data from API | "
            f"initial_prompt type: {type(initial_prompt)} | "
            f"initial_prompt value: {repr(initial_prompt)} | "
            f"transcription keys: {list(transcription.keys())}"
        )
        
        # Normaliser le prompt : convertir les chaînes vides en None
        if initial_prompt is not None and isinstance(initial_prompt, str) and initial_prompt.strip() == "":
            initial_prompt = None
        
        logger.info(
            f"[{transcription_id}] 🔍 DISTRIBUTED ORCHESTRATION | Initial prompt: {initial_prompt if initial_prompt else '(none)'}"
        )
        
        # ✅ NOUVEAU : Enregistrer le temps de début réel de l'orchestration
        # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
        processing_start_datetime = datetime.utcnow()
        
        # Mettre à jour le statut
        api_client.update_transcription(transcription_id, {
            "status": "processing",
            "worker_id": f"{config.instance_name}-orchestrator",
            "processing_start_time": processing_start_datetime.isoformat()  # ✅ NOUVEAU
        })
        
        # 1. Pré-traiter l'audio
        logger.info(f"[{transcription_id}] 🔧 DISTRIBUTED ORCHESTRATION | Step 1/6: Preprocessing audio...")
        preprocessed = preprocess_audio(file_path_obj, preserve_stereo_for_diarization=transcription.get('diarization_enabled', False))
        processed_path_mono = preprocessed['mono']
        
        # 2. Découper en segments (forcer la découpe car on est en mode distribué)
        logger.info(
            f"[{transcription_id}] ✂️ DISTRIBUTED ORCHESTRATION | Step 2/6: Splitting audio into segments | "
            f"VAD: {use_vad} | Segment length: {config.segment_length_ms}ms | "
            f"Force split: True (distributed mode)"
        )
        segment_paths = split_audio_intelligent(
            processed_path_mono,
            use_vad=use_vad,
            segment_length_ms=config.segment_length_ms,
            force_split_for_distribution=True  # Forcer la découpe en mode distribué
        )
        
        num_segments = len(segment_paths)
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 2/6: Segmentation complete | "
            f"Segments created: {num_segments} | "
            f"Will be distributed across available workers"
        )
        
        if num_segments == 0:
            raise ValueError("No segments created")
        
        # 3. Calculer les offsets temporels réels pour chaque segment de transcription
        # Ces offsets seront utilisés pour ajuster les timestamps des segments transcrits
        transcription_time_offsets = []
        current_offset = 0.0
        for i, seg_path in enumerate(segment_paths):
            transcription_time_offsets.append(current_offset)
            # Calculer la durée réelle du segment audio
            try:
                import soundfile as sf
                seg_duration = sf.info(str(seg_path)).duration
                current_offset += seg_duration
            except Exception as e:
                # Fallback : utiliser la durée moyenne estimée
                logger.warning(f"[{transcription_id}] ⚠️ Could not get segment {i} duration: {e}, using estimated duration")
                estimated_duration = get_audio_duration(file_path_obj) / num_segments
                current_offset += estimated_duration
        
        logger.info(
            f"[{transcription_id}] ⏱️ DISTRIBUTED ORCHESTRATION | Calculated time offsets for {num_segments} segments | "
            f"Total duration: {current_offset:.2f}s"
        )
        
        # 4. Préparer la diarisation distribuée si demandée
        use_diarization = transcription.get('diarization_enabled', False)
        diarization_segment_paths = []
        diarization_time_offsets = []
        
        if use_diarization:
            diarization_audio_path = preprocessed.get('stereo') if preprocessed.get('stereo') else processed_path_mono
            if diarization_audio_path and Path(diarization_audio_path).exists():
                # Vérifier le type de diarisation
                diarization_type = getattr(config, 'diarization_type', 'stereo')
                
                if diarization_type == 'stereo' and preprocessed.get('stereo'):
                    # Pour la diarisation stéréo : utiliser le fichier stéréo complet (une seule tâche)
                    # La diarisation stéréo est très rapide, pas besoin de découper
                    logger.info(f"[{transcription_id}] 🎤 DISTRIBUTED DIARIZATION | Using stereo diarization with full stereo file (no segmentation needed)")
                    diarization_segment_paths = [Path(preprocessed.get('stereo'))]
                    diarization_time_offsets = [0.0]  # Pas d'offset nécessaire pour un fichier complet
                else:
                    # Pour mono : utiliser les segments (comportement original)
                    logger.info(f"[{transcription_id}] 🎤 DISTRIBUTED DIARIZATION | Preparing diarization segments...")
                    diarization_segment_paths = segment_paths
                    # Calculer les offsets temporels pour chaque segment
                    current_offset = 0.0
                    for i, seg_path in enumerate(segment_paths):
                        diarization_time_offsets.append(current_offset)
                        # Estimer la durée du segment
                        try:
                            import soundfile as sf
                            seg_duration = sf.info(str(seg_path)).duration
                            current_offset += seg_duration
                        except:
                            # Fallback : utiliser la durée moyenne estimée
                            estimated_duration = get_audio_duration(file_path_obj) / num_segments
                            current_offset += estimated_duration
                logger.info(f"[{transcription_id}] 🎤 DISTRIBUTED DIARIZATION | Prepared {len(diarization_segment_paths)} segment(s) for diarization")
        
        # 5. Stocker les métadonnées dans Redis
        redis_manager = get_redis_manager()
        # ⚠️ IMPORTANT : Enregistrer le temps de début RÉEL de l'orchestration (pas de la création de la tâche)
        orchestration_start_time = time.time()
        
        segments_metadata = {
            "transcription_id": transcription_id,
            "total_segments": num_segments,
            "completed_segments": 0,
            "segment_paths": [str(p) for p in segment_paths],
            "transcription_time_offsets": transcription_time_offsets,  # Offsets temporels réels pour chaque segment
            "use_vad": use_vad,
            "use_diarization": use_diarization,
            "whisper_model": whisper_model,
            "initial_prompt": initial_prompt,  # ✅ NOUVEAU : Stocker l'initial_prompt pour tous les segments
            "processed_path_mono": str(processed_path_mono),
            "processed_path_stereo": str(preprocessed.get('stereo')) if preprocessed.get('stereo') else None,
            "is_stereo": preprocessed.get('is_stereo', False),
            "original_duration": get_audio_duration(file_path_obj),
            "orchestration_start_time": orchestration_start_time,
            "diarization_segment_paths": [str(p) for p in diarization_segment_paths],
            "diarization_time_offsets": diarization_time_offsets
        }
        
        # ⚠️ IMPORTANT : Calculer un TTL dynamique basé sur la durée estimée du traitement
        # TTL = durée audio * facteur de sécurité + marge
        # Pour éviter l'expiration pendant le traitement
        base_ttl = getattr(config, 'redis_transcription_ttl', 14400)  # TTL de base (4h par défaut)
        audio_duration = get_audio_duration(file_path_obj)
        # Estimer le temps de traitement : ~1.5x la durée audio (conservateur)
        estimated_processing_time = max(audio_duration * 1.5, 300)  # Minimum 5 minutes
        # TTL = temps estimé + marge de sécurité (2h)
        dynamic_ttl = int(estimated_processing_time + 7200)  # +2h de marge
        # Utiliser le maximum entre TTL de base et TTL dynamique
        ttl = max(base_ttl, dynamic_ttl)
        
        # ✅ NOUVEAU : Stocker le TTL original dans les métadonnées pour le monitoring
        segments_metadata["ttl_original"] = ttl
        
        logger.info(
            f"[{transcription_id}] 🔧 Redis TTL calculated | "
            f"Base TTL: {base_ttl}s | "
            f"Audio duration: {audio_duration:.1f}s | "
            f"Estimated processing: {estimated_processing_time:.1f}s | "
            f"Dynamic TTL: {dynamic_ttl}s | "
            f"Final TTL: {ttl}s ({ttl/3600:.1f}h)"
        )
        
        redis_manager.store_metadata(transcription_id, segments_metadata, ttl)
        redis_manager.reset_completed_count(transcription_id, ttl)  # Utiliser le TTL dynamique
        
        # ⚠️ IMPORTANT : Calculer un TTL dynamique basé sur la durée estimée du traitement
        # TTL = durée audio * facteur de sécurité + marge
        # Pour éviter l'expiration pendant le traitement
        base_ttl = getattr(config, 'redis_transcription_ttl', 14400)  # TTL de base (4h par défaut)
        audio_duration = get_audio_duration(file_path_obj)
        # Estimer le temps de traitement : ~1.5x la durée audio (conservateur)
        estimated_processing_time = max(audio_duration * 1.5, 300)  # Minimum 5 minutes
        # TTL = temps estimé + marge de sécurité (2h)
        dynamic_ttl = int(estimated_processing_time + 7200)  # +2h de marge
        # Utiliser le maximum entre TTL de base et TTL dynamique
        ttl = max(base_ttl, dynamic_ttl)
        
        logger.info(
            f"[{transcription_id}] 🔧 Redis TTL calculated | "
            f"Base TTL: {base_ttl}s | "
            f"Audio duration: {audio_duration:.1f}s | "
            f"Estimated processing: {estimated_processing_time:.1f}s | "
            f"Dynamic TTL: {dynamic_ttl}s | "
            f"Final TTL: {ttl}s ({ttl/3600:.1f}h)"
        )
        
        redis_manager.store_metadata(transcription_id, segments_metadata, ttl)
        redis_manager.reset_completed_count(transcription_id, ttl)  # Utiliser le TTL dynamique
        
        # Réinitialiser aussi le compteur de diarisation si nécessaire
        if use_diarization:
            redis_client = get_redis_client()
            diarization_counter_key = f"transcription:{transcription_id}:diarization_completed_count"
            redis_client.delete(diarization_counter_key)
            redis_client.set(diarization_counter_key, 0)
            redis_client.expire(diarization_counter_key, ttl)  # Utiliser le même TTL dynamique
            logger.debug(f"[{transcription_id}] 🔧 Diarization counter TTL set to {ttl}s")
        
        # 6. Créer une tâche pour chaque segment de transcription
        logger.info(
            f"[{transcription_id}] 📤 DISTRIBUTED ORCHESTRATION | Step 3/6: Creating transcription segment tasks | "
            f"Total segments: {num_segments} | "
            f"Queue: transcription | "
            f"Tasks will be distributed automatically by Celery"
        )
        segment_tasks = []
        from celery import current_app as celery_current_app
        
        for i, segment_path in enumerate(segment_paths):
            segment_task = celery_current_app.send_task(
                'transcribe_segment',
                args=[transcription_id, str(segment_path), i, num_segments],
                queue='transcription'
            )
            segment_tasks.append(segment_task.id)
            logger.info(
                f"[{transcription_id}] 📤 DISTRIBUTED ORCHESTRATION | Segment {i+1}/{num_segments} enqueued | "
                f"Task ID: {segment_task.id} | "
                f"File: {Path(segment_path).name} | "
                f"Waiting for available worker..."
            )
        
        # 7. Créer des tâches de diarisation distribuée si demandée
        diarization_tasks = []
        if use_diarization and diarization_segment_paths:
            logger.info(
                f"[{transcription_id}] 🎤 DISTRIBUTED ORCHESTRATION | Step 4/6: Creating diarization segment tasks | "
                f"Total segments: {len(diarization_segment_paths)} | "
                f"Queue: transcription | "
                f"Tasks will be distributed automatically by Celery"
            )
            
            for i, diarization_seg_path in enumerate(diarization_segment_paths):
                diarization_task = celery_current_app.send_task(
                    'diarize_segment',
                    args=[transcription_id, str(diarization_seg_path), i, len(diarization_segment_paths)],
                    queue='transcription'
                )
                diarization_tasks.append(diarization_task.id)
                logger.info(
                    f"[{transcription_id}] 🎤 DISTRIBUTED ORCHESTRATION | Diarization segment {i+1}/{len(diarization_segment_paths)} enqueued | "
                    f"Task ID: {diarization_task.id} | "
                    f"File: {Path(diarization_seg_path).name}"
                )
        
        # 8. Stocker les IDs des tâches
        redis_client = get_redis_client()
        tasks_key = f"transcription:{transcription_id}:segment_tasks"
        redis_client.setex(tasks_key, 3600, json.dumps(segment_tasks))
        
        if diarization_tasks:
            diarization_tasks_key = f"transcription:{transcription_id}:diarization_tasks"
            redis_client.setex(diarization_tasks_key, 3600, json.dumps(diarization_tasks))
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 6/6: All tasks created | "
            f"Transcription tasks: {num_segments} | "
            f"Diarization tasks: {len(diarization_tasks) if use_diarization else 0} | "
            f"Next: Workers will process segments in parallel"
        )
        
        return {
            "status": "orchestrated",
            "transcription_id": transcription_id,
            "num_segments": num_segments,
            "segment_tasks": segment_tasks
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Orchestration error: {e}", exc_info=True)
        api_client = None
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "status": "error",
                "error_message": f"Orchestration failed: {str(e)}"
            })
        except:
            pass
        
        # Supprimer le fichier .wav original même en cas d'échec d'orchestration
        try:
            if api_client:
                transcription = api_client.get_transcription(transcription_id)
                if transcription and transcription.get('file_path'):
                    original_file_path = transcription.get('file_path')
                    original_file_path_obj = Path(original_file_path)
                    if original_file_path_obj.exists() and original_file_path_obj.suffix.lower() == '.wav':
                        original_file_path_obj.unlink()
                        logger.info(f"[{transcription_id}] 🗑️ Original audio file deleted after orchestration error: {original_file_path}")
        except Exception as delete_error:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to delete original audio file after orchestration error: {delete_error}")
        
        raise

@celery_app.task(
    bind=True,
    name='transcribe_segment',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True
)
def transcribe_segment_task(self, transcription_id: str, segment_path: str, segment_index: int, total_segments: int):
    """
    Transcrit un seul segment audio.
    
    Args:
        transcription_id: ID de la transcription parente
        segment_path: Chemin vers le segment audio
        segment_index: Index du segment (0-based)
        total_segments: Nombre total de segments
    """
    logger.info(
        f"[{transcription_id}] 🎯 DISTRIBUTED SEGMENT STARTED | "
        f"Segment: {segment_index+1}/{total_segments} | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"File: {Path(segment_path).name}"
    )
    start_time = time.time()
    
    try:
        # Récupérer les métadonnées depuis Redis
        redis_manager = get_redis_manager()
        metadata = redis_manager.get_metadata(transcription_id)
        
        if not metadata:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        use_vad = metadata.get('use_vad', True)
        whisper_model = metadata.get('whisper_model', 'small')
        initial_prompt = metadata.get('initial_prompt')  # ✅ NOUVEAU : Récupérer l'initial_prompt depuis Redis
        
        # Normaliser le prompt : convertir les chaînes vides en None
        if initial_prompt is not None and isinstance(initial_prompt, str) and initial_prompt.strip() == "":
            initial_prompt = None
        
        # ⚠️ IMPORTANT : Dans la transcription distribuée, le prompt initial peut causer des suppressions de texte
        # car chaque segment est traité indépendamment. Le prompt initial de Whisper est conçu pour l'audio complet.
        # On utilise le prompt uniquement pour le premier segment (segment_index == 0) pour guider le contexte général,
        # mais pas pour les segments suivants pour éviter les suppressions de texte.
        use_prompt_for_segment = initial_prompt if (segment_index == 0) else None
        
        # ⚠️ IMPORTANT : Si un initial_prompt est fourni pour le premier segment, désactiver le VAD
        # Le VAD peut filtrer le début du premier segment même avec un initial_prompt
        use_vad_for_segment = use_vad if not use_prompt_for_segment else False
        
        logger.info(
            f"[{transcription_id}] ⚙️ DISTRIBUTED SEGMENT | Worker {config.instance_name} processing | "
            f"Segment: {segment_index+1}/{total_segments} | "
            f"Model: {whisper_model} | VAD: {use_vad_for_segment} | Initial prompt: {use_prompt_for_segment if use_prompt_for_segment else '(none - using only for first segment)'}"
        )
        
        if use_prompt_for_segment and not use_vad_for_segment:
            logger.info(f"[{transcription_id}] 🔍 VAD disabled for first segment with initial_prompt to ensure complete transcription from start")
        
        # Transcrit le segment
        transcription_service = get_transcription_service(model_name=whisper_model)
        segment_path_obj = Path(segment_path)
        
        if not segment_path_obj.exists():
            raise FileNotFoundError(f"Segment not found: {segment_path}")
        
        # ⚠️ IMPORTANT : Si c'est le premier segment avec initial_prompt, ajouter un padding de silence
        # pour forcer Whisper à traiter le début même s'il est silencieux
        padding_offset = 0.0
        final_segment_path = segment_path_obj
        if use_prompt_for_segment and segment_index == 0:
            try:
                audio = AudioSegment.from_file(str(segment_path_obj))
                # Ajouter 1 seconde de silence au début (16kHz mono)
                silence = AudioSegment.silent(duration=1000, frame_rate=16000)
                audio_with_padding = silence + audio
                padded_path = segment_path_obj.parent / f"{segment_path_obj.stem}_padded.wav"
                audio_with_padding.export(str(padded_path), format="wav")
                final_segment_path = padded_path
                padding_offset = 1.0  # 1 seconde de padding à compenser
                logger.info(f"[{transcription_id}] 🔍 Added 1s silence padding at start of first segment to force Whisper to process beginning (will adjust timestamps by -1s)")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Could not add silence padding to first segment: {e}, using original segment")
        
        text, segments_list, lang = transcription_service.transcribe_segment(
            final_segment_path,
            use_vad=use_vad_for_segment,  # VAD désactivé pour le premier segment si initial_prompt
            initial_prompt=use_prompt_for_segment,  # ✅ Utiliser le prompt uniquement pour le premier segment
            padding_offset=padding_offset  # Compenser le padding de silence
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Récupérer l'offset temporel réel depuis les métadonnées
        # Cet offset correspond à la position réelle du segment dans l'audio complet
        transcription_time_offsets = metadata.get('transcription_time_offsets', [])
        if segment_index < len(transcription_time_offsets):
            time_offset = transcription_time_offsets[segment_index]
        else:
            # Fallback : calculer l'offset à partir des segments précédents (ancienne méthode)
            logger.warning(
                f"[{transcription_id}] ⚠️ DISTRIBUTED SEGMENT | "
                f"Time offset not found in metadata for segment {segment_index}, using fallback calculation"
            )
            time_offset = 0.0
            if segment_index > 0:
                for prev_idx in range(segment_index):
                    prev_result = redis_manager.get_segment_result(transcription_id, prev_idx)
                    if prev_result and prev_result.get('segments'):
                        last_segment = prev_result['segments'][-1]
                        time_offset = last_segment.get('end', 0.0)
                        break
        
        logger.debug(
            f"[{transcription_id}] ⏱️ DISTRIBUTED SEGMENT | "
            f"Segment {segment_index+1}/{total_segments} | "
            f"Time offset: {time_offset:.2f}s"
        )
        
        # Ajuster les timestamps avec l'offset réel
        adjusted_segments = []
        for seg in segments_list:
            adjusted_segments.append({
                "start": round(seg["start"] + time_offset, 2),
                "end": round(seg["end"] + time_offset, 2),
                "text": seg["text"]
            })
        
        # Stocker le résultat dans Redis
        result = {
            "segment_index": segment_index,
            "text": text,
            "segments": adjusted_segments,
            "language": lang,
            "processing_time": processing_time,
            "time_offset": time_offset
        }
        
        # Stocker le résultat et incrémenter le compteur
        base_ttl = getattr(config, 'redis_transcription_ttl', 14400)
        redis_manager.store_segment_result(transcription_id, segment_index, result, base_ttl)
        
        # ⚠️ IMPORTANT : Incrémenter le compteur AVANT de calculer le TTL dynamique
        # Cela garantit que completed_count est défini avant utilisation
        completed_count = redis_manager.increment_completed_count(transcription_id)
        
        # ⚠️ IMPORTANT : Calculer un TTL dynamique basé sur le temps restant estimé
        # Cela évite l'expiration si le traitement prend plus de temps que prévu
        # Calculer après avoir incrémenté le compteur pour connaître le nombre de segments restants
        if 'orchestration_start_time' in metadata:
            elapsed_time = time.time() - metadata['orchestration_start_time']
            # completed_count inclut déjà le segment actuel, donc pas besoin de -1
            remaining_segments = max(0, metadata.get('total_segments', 1) - completed_count)
            # Estimer le temps restant : temps moyen par segment * segments restants
            avg_time_per_segment = processing_time  # Utiliser le temps actuel comme estimation
            estimated_remaining_time = max(avg_time_per_segment * remaining_segments, 300)  # Minimum 5 min
            # TTL = temps restant estimé + marge de sécurité (1h)
            dynamic_ttl = int(estimated_remaining_time + 3600)
            ttl = max(base_ttl, dynamic_ttl)
        else:
            ttl = base_ttl
        
        # ✅ NOUVEAU : Vérifier la santé du TTL avant de continuer
        ttl_health = redis_manager.check_ttl_health(transcription_id, threshold_percent=0.2)
        
        if ttl_health['alert']:
            logger.warning(
                f"[{transcription_id}] ⚠️ TTL ALERT | "
                f"Remaining: {ttl_health['ttl_remaining']}s ({ttl_health['percent_remaining']:.1f}%) | "
                f"Action: Renewing TTL to {ttl}s"
            )
        elif ttl_health['percent_remaining'] < 50.0:  # 50% restant
            logger.info(
                f"[{transcription_id}] ℹ️ TTL WARNING | "
                f"Remaining: {ttl_health['ttl_remaining']}s ({ttl_health['percent_remaining']:.1f}%)"
            )
        
        # Mettre à jour les métadonnées avec TTL renouvelé
        metadata['completed_segments'] = completed_count
        redis_manager.store_metadata(transcription_id, metadata, ttl)
        
        # ⚠️ IMPORTANT : Renouveler le TTL de toutes les clés Redis associées à cette transcription
        # Cela évite l'expiration si le traitement prend plus de temps que prévu
        redis_manager.refresh_ttl(transcription_id, ttl, total_segments)
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED SEGMENT COMPLETED | "
            f"Segment: {segment_index+1}/{total_segments} | "
            f"Worker: {config.instance_name} | "
            f"Processing time: {processing_time}s | "
            f"Progress: {completed_count}/{total_segments} segments done ({100*completed_count/total_segments:.1f}%)"
        )
        
        # Si tous les segments sont terminés, vérifier aussi la diarisation avant d'agréger
        if completed_count >= total_segments:
            # Vérifier si la diarisation est aussi terminée (si demandée)
            use_diarization = metadata.get('use_diarization', False)
            diarization_ready = True
            
            if use_diarization:
                diarization_segment_paths = metadata.get('diarization_segment_paths', [])
                if diarization_segment_paths:
                    redis_client = get_redis_client()
                    diarization_counter_key = f"transcription:{transcription_id}:diarization_completed_count"
                    diarization_completed = int(redis_client.get(diarization_counter_key) or 0)
                    diarization_ready = diarization_completed >= len(diarization_segment_paths)
                    
                    if not diarization_ready:
                        logger.info(
                            f"[{transcription_id}] ⏳ Waiting for diarization: {diarization_completed}/{len(diarization_segment_paths)} segments done"
                        )
            
            if diarization_ready and redis_manager.acquire_aggregation_lock(transcription_id):
                logger.info(
                    f"[{transcription_id}] 🎉 DISTRIBUTED MODE | All segments completed | "
                    f"Total: {total_segments} segments | "
                    f"Diarization: {'ready' if use_diarization else 'N/A'} | "
                    f"All workers finished | "
                    f"Triggering aggregation... (lock acquired by {config.instance_name})"
                )
                from celery import current_app as celery_current_app
                aggregate_task = celery_current_app.send_task(
                    'aggregate_segments',
                    args=[transcription_id],
                    queue='transcription',
                    countdown=1
                )
                logger.info(
                    f"[{transcription_id}] ✅ DISTRIBUTED MODE | Aggregation task enqueued | "
                    f"Task ID: {aggregate_task.id} | "
                    f"Queue: transcription | "
                    f"Next: Reassembling all segments"
                )
            elif not diarization_ready:
                logger.info(
                    f"[{transcription_id}] ℹ️ DISTRIBUTED MODE | Transcription complete but waiting for diarization"
                )
            else:
                logger.info(
                    f"[{transcription_id}] ℹ️ DISTRIBUTED MODE | All segments completed but aggregation already triggered by another worker"
                )
        
        return {
            "status": "success",
            "segment_index": segment_index,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Segment {segment_index+1} error: {e}", exc_info=True)
        raise

@celery_app.task(
    bind=True,
    name='diarize_segment',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True
)
def diarize_segment_task(self, transcription_id: str, segment_path: str, segment_index: int, total_segments: int):
    """
    Diarise un seul segment audio.
    
    Args:
        transcription_id: ID de la transcription parente
        segment_path: Chemin vers le segment audio
        segment_index: Index du segment (0-based)
        total_segments: Nombre total de segments
    """
    logger.info(
        f"[{transcription_id}] 🎤 DISTRIBUTED DIARIZATION SEGMENT STARTED | "
        f"Segment: {segment_index+1}/{total_segments} | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"File: {Path(segment_path).name}"
    )
    start_time = time.time()
    
    try:
        # Récupérer les métadonnées depuis Redis
        redis_manager = get_redis_manager()
        metadata = redis_manager.get_metadata(transcription_id)
        
        if not metadata:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        whisper_model = metadata.get('whisper_model', 'small')
        
        # Obtenir le service de transcription et charger la diarisation
        transcription_service = get_transcription_service(model_name=whisper_model)
        
        if transcription_service.diarization_service is None:
            logger.info(f"[{transcription_id}] 🔄 Loading diarization service (lazy loading)...")
            transcription_service._load_diarization_service()
        
        if not transcription_service.diarization_service or not transcription_service.diarization_service.pipeline:
            raise ValueError("Diarization service not available")
        
        segment_path_obj = Path(segment_path)
        if not segment_path_obj.exists():
            raise FileNotFoundError(f"Segment not found: {segment_path}")
        
        # Diariser le segment
        diarization_segments = transcription_service.diarization_service.diarize(segment_path_obj)
        
        processing_time = round(time.time() - start_time, 2)
        
        # Récupérer l'offset temporel depuis les métadonnées
        diarization_time_offsets = metadata.get('diarization_time_offsets', [])
        time_offset = diarization_time_offsets[segment_index] if segment_index < len(diarization_time_offsets) else 0.0
        
        # Stocker le résultat dans Redis
        result = {
            "segment_index": segment_index,
            "segments": diarization_segments,
            "processing_time": processing_time,
            "time_offset": time_offset
        }
        
        # Stocker le résultat et incrémenter le compteur
        base_ttl = getattr(config, 'redis_transcription_ttl', 14400)
        redis_manager.store_diarization_result(transcription_id, segment_index, result, base_ttl)
        
        # ⚠️ IMPORTANT : Incrémenter le compteur AVANT de calculer le TTL dynamique
        # Cela garantit que completed_count est défini avant utilisation
        completed_count = redis_manager.increment_diarization_count(transcription_id)
        
        # ⚠️ IMPORTANT : Calculer un TTL dynamique basé sur le temps restant estimé
        # Calculer après avoir incrémenté le compteur pour connaître le nombre de segments restants
        diarization_segment_paths = metadata.get('diarization_segment_paths', [])
        total_diarization_segments = len(diarization_segment_paths) if diarization_segment_paths else 0
        if 'orchestration_start_time' in metadata and total_diarization_segments > 0:
            elapsed_time = time.time() - metadata['orchestration_start_time']
            # completed_count inclut déjà le segment actuel, donc pas besoin de -1
            remaining_segments = max(0, total_diarization_segments - completed_count)
            avg_time_per_segment = processing_time
            estimated_remaining_time = max(avg_time_per_segment * remaining_segments, 300)
            dynamic_ttl = int(estimated_remaining_time + 3600)
            ttl = max(base_ttl, dynamic_ttl)
        else:
            ttl = base_ttl
        
        # ✅ NOUVEAU : Vérifier la santé du TTL avant de continuer
        ttl_health = redis_manager.check_ttl_health(transcription_id, threshold_percent=0.2)
        
        if ttl_health['alert']:
            logger.warning(
                f"[{transcription_id}] ⚠️ TTL ALERT (diarization) | "
                f"Remaining: {ttl_health['ttl_remaining']}s ({ttl_health['percent_remaining']:.1f}%) | "
                f"Action: Renewing TTL to {ttl}s"
            )
        elif ttl_health['percent_remaining'] < 50.0:  # 50% restant
            logger.info(
                f"[{transcription_id}] ℹ️ TTL WARNING (diarization) | "
                f"Remaining: {ttl_health['ttl_remaining']}s ({ttl_health['percent_remaining']:.1f}%)"
            )
        
        # ⚠️ IMPORTANT : Renouveler le TTL de toutes les clés Redis associées à cette transcription
        redis_manager.refresh_ttl(transcription_id, ttl, total_diarization_segments)
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED DIARIZATION SEGMENT COMPLETED | "
            f"Segment: {segment_index+1}/{total_segments} | "
            f"Worker: {config.instance_name} | "
            f"Processing time: {processing_time}s | "
            f"Segments: {len(diarization_segments)} | "
            f"Progress: {completed_count}/{total_segments} segments done ({100*completed_count/total_segments:.1f}%)"
        )
        
        # Si tous les segments de diarisation sont terminés, vérifier si on peut déclencher l'agrégation
        if completed_count >= total_segments:
            # Vérifier si la transcription est aussi terminée
            redis_client = get_redis_client()
            transcription_counter_key = f"transcription:{transcription_id}:completed_count"
            transcription_completed = int(redis_client.get(transcription_counter_key) or 0)
            total_transcription_segments = metadata.get('total_segments', 0)
            
            if transcription_completed >= total_transcription_segments:
                # Les deux sont terminés, déclencher l'agrégation
                if redis_manager.acquire_aggregation_lock(transcription_id):
                    logger.info(
                        f"[{transcription_id}] 🎉 DISTRIBUTED MODE | All transcription and diarization segments completed | "
                        f"Triggering aggregation... (lock acquired by {config.instance_name})"
                    )
                    from celery import current_app as celery_current_app
                    aggregate_task = celery_current_app.send_task(
                        'aggregate_segments',
                        args=[transcription_id],
                        queue='transcription',
                        countdown=1
                    )
                    logger.info(
                        f"[{transcription_id}] ✅ DISTRIBUTED MODE | Aggregation task enqueued | "
                        f"Task ID: {aggregate_task.id}"
                    )
        
        return {
            "status": "success",
            "segment_index": segment_index,
            "processing_time": processing_time,
            "segments_count": len(diarization_segments)
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Diarization segment {segment_index+1} error: {e}", exc_info=True)
        raise

@celery_app.task(
    bind=True,
    name='aggregate_segments',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True
)
def aggregate_segments_task(self, transcription_id: str):
    """
    Réassemble les segments transcrits en un résultat final.
    
    Args:
        transcription_id: ID de la transcription
    """
    logger.info(
        f"[{transcription_id}] 🔗 DISTRIBUTED AGGREGATION STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"Will reassemble all completed segments"
    )
    start_time = time.time()
    
    def get_segment_order_key(seg):
        """
        Calcule une clé d'ordre pour garantir un ordre chronologique correct.
        Utilise start comme critère principal, puis end comme tie-breaker.
        Cela garantit un ordre stable même avec des chevauchements.
        """
        start = seg.get('start', 0)
        end = seg.get('end', start)
        # Utiliser start comme critère principal, puis end comme tie-breaker
        # Cela garantit un ordre chronologique correct même avec des chevauchements
        return (start, end)
    
    def filter_overlapping_segments(segments, early_segment_threshold=2.0, overlap_threshold=0.5, start_window=0.5):
        """
        Filtre les segments qui commencent très tôt (artefacts du début) 
        et qui se chevauchent significativement avec d'autres segments.
        
        Amélioration : Compare tous les segments entre eux, pas seulement early vs later,
        pour mieux détecter les artefacts qui commencent presque en même temps.
        
        Args:
            segments: Liste de segments
            early_segment_threshold: Seuil en secondes. Les segments qui commencent avant ce seuil
                                    sont considérés comme "début" et peuvent être filtrés.
            overlap_threshold: Seuil de chevauchement (0.0-1.0). Un segment est supprimé
                              s'il se chevauche à plus de ce seuil avec un autre segment.
            start_window: Fenêtre temporelle en secondes. Les segments qui commencent dans cette
                          fenêtre sont comparés entre eux pour détecter les doublons.
        
        Returns:
            Liste de segments filtrés
        """
        if not segments:
            return segments
        
        # Trier par start
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        segments_to_remove = []
        
        # Identifier les segments qui commencent très tôt (candidats pour être des artefacts)
        early_segments = [seg for seg in sorted_segments if seg.get('start', 0) < early_segment_threshold]
        
        # Pour chaque segment du début, comparer avec tous les autres segments
        for i, early_seg in enumerate(early_segments):
            if early_seg in segments_to_remove:
                continue
                
            early_start = early_seg.get('start', 0)
            early_end = early_seg.get('end', early_start)
            early_duration = early_end - early_start
            
            if early_duration <= 0:
                continue
            
            # Comparer avec tous les autres segments (y compris ceux qui commencent aussi tôt)
            for j, other_seg in enumerate(sorted_segments):
                if i == j or other_seg in segments_to_remove:
                    continue
                    
                other_start = other_seg.get('start', 0)
                other_end = other_seg.get('end', other_start)
                other_duration = other_end - other_start
                
                if other_duration <= 0:
                    continue
                
                # Calculer le chevauchement
                overlap_start = max(early_start, other_start)
                overlap_end = min(early_end, other_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration <= 0:
                    continue  # Pas de chevauchement
                
                # Calculer le ratio de chevauchement par rapport au segment du début
                overlap_ratio = overlap_duration / early_duration if early_duration > 0 else 0
                
                # Si les deux segments commencent dans une fenêtre très courte (artefacts du début)
                # et se chevauchent significativement, supprimer celui qui commence le plus tôt
                # ou celui qui est le plus court
                start_diff = abs(early_start - other_start)
                
                if start_diff <= start_window and overlap_ratio >= overlap_threshold:
                    # Décider lequel supprimer : celui qui commence le plus tôt, ou le plus court si même start
                    if early_start < other_start:
                        # Le segment early commence plus tôt, le supprimer
                        segments_to_remove.append(early_seg)
                        break
                    elif early_start == other_start:
                        # Même start : supprimer le plus court
                        if early_duration < other_duration:
                            segments_to_remove.append(early_seg)
                            break
                        elif early_duration == other_duration:
                            # Même durée : supprimer celui qui se termine le plus tôt (artefact probable)
                            if early_end < other_end:
                                segments_to_remove.append(early_seg)
                                break
                elif overlap_ratio >= overlap_threshold:
                    # Chevauchement significatif même si pas dans la fenêtre de début
                    # Supprimer celui qui commence le plus tôt
                    if early_start < other_start:
                        segments_to_remove.append(early_seg)
                        break
        
        # Filtrer : garder tous les segments sauf ceux identifiés comme artefacts
        filtered = [seg for seg in sorted_segments if seg not in segments_to_remove]
        
        return filtered
    
    try:
        redis_manager = get_redis_manager()
        metadata = redis_manager.get_metadata(transcription_id)
        
        # ✅ NOUVEAU : Vérifier la santé du TTL avant l'agrégation
        if metadata:
            ttl_health = redis_manager.check_ttl_health(transcription_id, threshold_percent=0.2)
            if not ttl_health['healthy']:
                logger.error(
                    f"[{transcription_id}] ❌ TTL EXPIRED | "
                    f"Cannot aggregate segments - data may be lost | "
                    f"TTL remaining: {ttl_health['ttl_remaining']}s"
                )
                # Ne pas lever d'exception, mais logger l'erreur
        
        if not metadata:
            # Cas particulier : métadonnées manquantes (souvent lié à un TTL expiré ou à un reset de Redis)
            error_message = f"Metadata not found for transcription {transcription_id}"
            logger.error(
                f"[{transcription_id}] ❌ Aggregation error (missing metadata): {error_message}"
            )
            try:
                api_client = get_api_client()
                api_client.update_transcription(transcription_id, {
                    "status": "error",
                    "error_message": error_message
                })
            except Exception as update_err:
                logger.error(
                    f"[{transcription_id}] Failed to update transcription status after missing metadata: {update_err}",
                    exc_info=True
                )
            # Ne pas lever d'exception ici pour éviter de laisser la tâche bloquée en retry
            return {
                "status": "error",
                "transcription_id": transcription_id,
                "error": error_message
            }
        
        total_segments = metadata['total_segments']
        use_diarization = metadata.get('use_diarization', False)
        
        # Récupérer tous les résultats des segments
        logger.info(
            f"[{transcription_id}] 📥 DISTRIBUTED AGGREGATION | Step 1/3: Collecting segment results | "
            f"Expected segments: {total_segments}"
        )
        all_segments = []
        full_text = ""
        language_detected = None
        max_segment_time = 0.0
        segment_processing_times = []
        
        for i in range(total_segments):
            result = redis_manager.get_segment_result(transcription_id, i)
            if not result:
                raise ValueError(f"Result not found for segment {i} of transcription {transcription_id}")
            
            all_segments.extend(result['segments'])
            full_text += result['text'] + " "
            segment_time = result.get('processing_time', 0.0)
            segment_processing_times.append(segment_time)
            max_segment_time = max(max_segment_time, segment_time)
            
            if language_detected is None:
                language_detected = result.get('language')
        
        # Calculer le temps réel écoulé depuis le début de l'orchestration
        orchestration_start_time = metadata.get('orchestration_start_time')
        if orchestration_start_time:
            real_elapsed_time = round(time.time() - orchestration_start_time, 2)
        else:
            # Fallback : utiliser le temps max des segments + temps d'agrégation
            real_elapsed_time = max_segment_time
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED AGGREGATION | Step 1/3: All results collected | "
            f"Segments: {len(all_segments)} | "
            f"Max segment time: {max_segment_time:.1f}s (parallel) | "
            f"Real elapsed time: {real_elapsed_time:.1f}s"
        )
        
        # Filtrer les segments qui commencent très tôt (artefacts du début)
        # et qui se chevauchent significativement avec d'autres segments
        # Cette approche compare tous les segments entre eux pour mieux détecter les doublons
        original_count = len(all_segments)
        all_segments = filter_overlapping_segments(all_segments, early_segment_threshold=2.0, overlap_threshold=0.5, start_window=0.5)
        filtered_count = len(all_segments)
        if original_count != filtered_count:
            logger.info(
                f"[{transcription_id}] 🔍 DISTRIBUTED AGGREGATION | Filtered overlapping segments | "
                f"Before: {original_count} | After: {filtered_count} | Removed: {original_count - filtered_count}"
            )
        
        # Trier les segments par ordre chronologique (start, puis end)
        # ⚠️ IMPORTANT: Utiliser start comme critère principal garantit un ordre chronologique correct
        # même avec des chevauchements. Le tri par start/end est plus fiable que le point médian.
        all_segments.sort(key=get_segment_order_key)
        logger.info(
            f"[{transcription_id}] 🔄 DISTRIBUTED AGGREGATION | Step 2/3: Segments sorted chronologically (start, end) | "
            f"Total segments: {len(all_segments)}"
        )
        
        # Diarisation si demandée (distribuée ou non)
        if use_diarization:
            logger.info(f"[{transcription_id}] 🎤 Running speaker diarization on aggregated segments...")
            try:
                # Vérifier si la diarisation distribuée a été utilisée
                diarization_segment_paths = metadata.get('diarization_segment_paths', [])
                diarization_time_offsets = metadata.get('diarization_time_offsets', [])
                
                if diarization_segment_paths and len(diarization_segment_paths) > 0:
                    # Mode distribué : récupérer les résultats de diarisation depuis Redis
                    logger.info(f"[{transcription_id}] 🎤 DISTRIBUTED DIARIZATION | Collecting diarization results from {len(diarization_segment_paths)} segments...")
                    
                    diarization_results = []
                    for i in range(len(diarization_segment_paths)):
                        diarization_result = redis_manager.get_diarization_result(transcription_id, i)
                        if diarization_result:
                            diarization_results.append(diarization_result.get('segments', []))
                        else:
                            logger.warning(f"[{transcription_id}] ⚠️ Diarization result not found for segment {i}")
                            diarization_results.append([])
                    
                    # Fusionner les résultats de diarisation
                    if any(diarization_results):
                        merger = DiarizationMerger()
                        merged_diarization_segments = merger.merge_diarization_segments(
                            diarization_results,
                            diarization_time_offsets
                        )
                        
                        if merged_diarization_segments:
                            transcription_service = get_transcription_service(model_name=metadata.get('whisper_model', 'small'))
                            
                            # Charger le service de diarisation si nécessaire (lazy loading)
                            if transcription_service.diarization_service is None:
                                logger.info(f"[{transcription_id}] 🔄 Loading diarization service (lazy loading)...")
                                transcription_service._load_diarization_service()
                            
                            if transcription_service.diarization_service and transcription_service.diarization_service.pipeline:
                                all_segments = transcription_service.diarization_service.assign_speakers_to_segments(
                                    all_segments,
                                    merged_diarization_segments
                                )
                                # Re-trier après l'assignation des speakers pour garantir l'ordre chronologique
                                all_segments.sort(key=get_segment_order_key)
                                logger.info(f"[{transcription_id}] ✅ DISTRIBUTED DIARIZATION | Completed and assigned to segments | Re-sorted chronologically")
                            else:
                                logger.warning(f"[{transcription_id}] ⚠️ Diarization service not available after loading")
                        else:
                            logger.warning(f"[{transcription_id}] ⚠️ No diarization segments after merging")
                    else:
                        logger.warning(f"[{transcription_id}] ⚠️ No diarization results found")
                else:
                    # Mode non distribué : diarisation sur l'audio complet (fallback)
                    logger.info(f"[{transcription_id}] 🎤 Using non-distributed diarization (fallback)...")
                    transcription_service = get_transcription_service(model_name=metadata.get('whisper_model', 'small'))
                    
                    if transcription_service.diarization_service is None:
                        logger.info(f"[{transcription_id}] 🔄 Loading diarization service (lazy loading)...")
                        transcription_service._load_diarization_service()
                    
                    if transcription_service.diarization_service and transcription_service.diarization_service.pipeline:
                        diarization_audio_path = Path(metadata['processed_path_stereo']) if metadata.get('processed_path_stereo') else Path(metadata['processed_path_mono'])
                        
                        if diarization_audio_path.exists():
                            logger.info(f"[{transcription_id}] 🎯 Using {'STEREO' if metadata.get('processed_path_stereo') else 'MONO'} audio for diarization")
                            diarization_segments = transcription_service.diarization_service.diarize(diarization_audio_path)
                            
                            if diarization_segments:
                                all_segments = transcription_service.diarization_service.assign_speakers_to_segments(
                                    all_segments,
                                    diarization_segments
                                )
                                # Re-trier après l'assignation des speakers pour garantir l'ordre chronologique
                                all_segments.sort(key=get_segment_order_key)
                                logger.info(f"[{transcription_id}] ✅ Diarization completed and assigned to segments | Re-sorted chronologically")
                            else:
                                logger.warning(f"[{transcription_id}] ⚠️ Diarization returned no segments")
                        else:
                            logger.warning(f"[{transcription_id}] ⚠️ Diarization audio file not found: {diarization_audio_path}")
                    else:
                        logger.warning(f"[{transcription_id}] ⚠️ Diarization requested but service not available (check diarization configuration)")
            except Exception as e:
                logger.error(f"[{transcription_id}] ❌ Diarization error: {e}", exc_info=True)
        
        # ⚠️ IMPORTANT : Reconstruire le texte final APRÈS toutes les modifications (tri + diarisation)
        # pour garantir que le texte correspond exactement à l'ordre chronologique des segments
        # et que chaque segment avec son texte est pris en compte dans l'ordre correct
        
        # Re-trier par ordre chronologique (au cas où la diarisation aurait modifié l'ordre)
        all_segments.sort(key=get_segment_order_key)
        
        # Construire le texte final en joignant les segments dans l'ordre chronologique
        full_text = " ".join(seg.get('text', '') for seg in all_segments if seg.get('text', '').strip())
        
        # Sauvegarder le résultat final
        api_client = get_api_client()
        transcription = api_client.get_transcription(transcription_id)
        enrichment_requested = transcription.get('enrichment_requested', False) if transcription else False
        
        status = "transcribed" if enrichment_requested else "done"
        aggregation_time = round(time.time() - start_time, 2)
        
        # ⚠️ IMPORTANT : Calculer uniquement le temps de traitement RÉEL (sans temps d'attente)
        # Le temps réel est le temps écoulé depuis le début de l'orchestration (quand le worker a commencé)
        # Cela exclut le temps d'attente dans la file Celery
        if orchestration_start_time:
            processing_end_time_float = time.time()
            total_processing_time = round(processing_end_time_float - orchestration_start_time, 2)
        else:
            # Fallback : max segment + agrégation (approximation)
            total_processing_time = round(max_segment_time + aggregation_time, 2)
        
        # ✅ NOUVEAU : Calculer les métriques de performance
        queued_at = transcription.get('queued_at') if transcription else None
        processing_start_time_str = transcription.get('processing_start_time') if transcription else None
        
        # Calculer le temps d'attente dans la file
        queue_wait_time = None
        if queued_at and processing_start_time_str:
            try:
                # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
                queued_dt = datetime.fromisoformat(queued_at.replace('Z', '+00:00'))
                start_dt = datetime.fromisoformat(processing_start_time_str.replace('Z', '+00:00'))
                queue_wait_time = round((start_dt - queued_dt).total_seconds(), 2)
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to calculate queue_wait_time: {e}")
        
        # ✅ datetime est déjà importé au niveau du module, on peut l'utiliser directement
        processing_end_datetime = datetime.utcnow()
        
        api_client.update_transcription(transcription_id, {
            "status": status,
            "text": full_text.strip(),
            "segments": json.dumps(all_segments),
            "language": language_detected,
            "duration": metadata.get('original_duration', 0.0),
            "processing_time": total_processing_time,
            "segments_count": len(all_segments),
            "queue_wait_time": queue_wait_time,  # ✅ NOUVEAU
            "processing_end_time": processing_end_datetime.isoformat()  # ✅ NOUVEAU
        })
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED AGGREGATION | Step 3/3: Aggregation completed | "
            f"Total segments: {len(all_segments)} | "
            f"Real processing time: {total_processing_time:.1f}s (from orchestration start) | "
            f"Max segment time: {max_segment_time:.1f}s | "
            f"Aggregation time: {aggregation_time:.1f}s | "
            f"Status: {status} | "
            f"Result saved to database"
        )
        
        # Supprimer le fichier .wav original après transcription réussie
        if transcription and transcription.get('file_path'):
            original_file_path = transcription.get('file_path')
            try:
                original_file_path_obj = Path(original_file_path)
                if original_file_path_obj.exists() and original_file_path_obj.suffix.lower() == '.wav':
                    original_file_path_obj.unlink()
                    logger.info(f"[{transcription_id}] 🗑️ Original audio file deleted: {original_file_path}")
                elif original_file_path_obj.exists():
                    logger.debug(f"[{transcription_id}] ℹ️ File not deleted (not .wav): {original_file_path}")
            except Exception as delete_error:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to delete original audio file: {delete_error}")
        
        # Nettoyer les données Redis (transcription + diarisation)
        try:
            redis_manager.cleanup(transcription_id, total_segments)
            
            # Nettoyer aussi les résultats de diarisation distribuée
            if use_diarization:
                diarization_segment_paths = metadata.get('diarization_segment_paths', [])
                if diarization_segment_paths:
                    redis_client = get_redis_client()
                    pipe = redis_client.pipeline()
                    for i in range(len(diarization_segment_paths)):
                        pipe.delete(f"transcription:{transcription_id}:diarization:{i}:result")
                    pipe.delete(f"transcription:{transcription_id}:diarization_tasks")
                    pipe.delete(f"transcription:{transcription_id}:diarization_completed_count")
                    pipe.execute()
            
            # Nettoyer les fichiers temporaires
            from audio_utils import cleanup_segments
            segment_paths = [Path(p) for p in metadata.get('segment_paths', [])]
            cleanup_segments(segment_paths)
            
            if metadata.get('processed_path_mono'):
                mono_path = Path(metadata['processed_path_mono'])
                if mono_path.exists() and mono_path != Path(transcription.get('file_path', '')):
                    mono_path.unlink()
            
            if metadata.get('processed_path_stereo'):
                stereo_path = Path(metadata['processed_path_stereo'])
                if stereo_path.exists():
                    stereo_path.unlink()
        except Exception as cleanup_error:
            logger.warning(f"[{transcription_id}] ⚠️ Cleanup error: {cleanup_error}")
        
        # Déclencher l'enrichissement si demandé
        if enrichment_requested:
            trigger_enrichment_task(transcription_id, api_client)
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "segments_count": len(all_segments),
            "total_processing_time": total_processing_time
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Aggregation error: {e}", exc_info=True)
        api_client = None
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "status": "error",
                "error_message": f"Aggregation failed: {str(e)}"
            })
        except:
            pass
        
        # Supprimer le fichier .wav original même en cas d'échec
        try:
            if api_client:
                transcription = api_client.get_transcription(transcription_id)
                if transcription and transcription.get('file_path'):
                    original_file_path = transcription.get('file_path')
                    original_file_path_obj = Path(original_file_path)
                    if original_file_path_obj.exists() and original_file_path_obj.suffix.lower() == '.wav':
                        original_file_path_obj.unlink()
                        logger.info(f"[{transcription_id}] 🗑️ Original audio file deleted after aggregation error: {original_file_path}")
        except Exception as delete_error:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to delete original audio file after aggregation error: {delete_error}")
        
        raise

if __name__ == "__main__":
    # ... (Le reste de votre fichier __main__ est parfait et n'a pas besoin d'être modifié) ...
    logger.info(f"🚀 Starting Celery worker: {config.instance_name}")
    celery_app.worker_main([
        'worker',
        f'--loglevel={config.log_level.lower()}',
        f'--concurrency={config.max_workers}',
        f'--hostname={config.instance_name}@%h'
    ])