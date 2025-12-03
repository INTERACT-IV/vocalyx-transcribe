"""
vocalyx-transcribe/worker.py
Worker Celery pour la transcription audio
"""

import logging
import time
import os
import psutil
import threading
import json
import redis
import gzip
import base64
from pathlib import Path
from datetime import datetime
from celery.signals import worker_init
from celery.worker.control import Panel

from celery import Celery
from config import Config
from transcription_service import TranscriptionService  # Compatibilité
from api_client import VocalyxAPIClient  # Compatibilité
from infrastructure.api.api_client import VocalyxAPIClient as VocalyxAPIClientRefactored
from application.services.transcription_worker_service import TranscriptionWorkerService
from audio_utils import split_audio_intelligent, preprocess_audio, get_audio_duration

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

# --- MODIFICATION 2 : Déclarer les services comme 'None' ---
_api_client = None
_transcription_service = None
_redis_client = None

def get_redis_client():
    """Obtient un client Redis pour stocker les résultats des segments"""
    global _redis_client
    if _redis_client is None:
        # Utiliser la DB Redis dédiée pour les transcriptions (isolation des données)
        redis_url = getattr(config, 'redis_transcription_url', None)
        if not redis_url:
            # Fallback : utiliser DB 2 de la même instance
            base_url = config.celery_broker_url.rsplit('/', 1)[0]  # Enlever /0
            redis_url = f"{base_url}/2"
        
        logger.info(f"🔌 Initializing Redis transcription client: {redis_url}")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test de connexion
        try:
            _redis_client.ping()
            logger.info(f"✅ Redis transcription client connected successfully: {redis_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis transcription: {redis_url} - {e}")
            raise
    
    return _redis_client

def _compress_json(data: dict) -> str:
    """Compresse un dictionnaire JSON pour économiser la mémoire Redis"""
    if not getattr(config, 'redis_transcription_compress', True):
        return json.dumps(data)
    
    json_str = json.dumps(data)
    compressed = gzip.compress(json_str.encode('utf-8'), compresslevel=6)
    # Encoder en base64 pour stockage Redis (string)
    return base64.b64encode(compressed).decode('utf-8')

def _decompress_json(compressed_str: str) -> dict:
    """Décompresse une chaîne JSON compressée"""
    if not getattr(config, 'redis_transcription_compress', True):
        return json.loads(compressed_str)
    
    try:
        # Décoder depuis base64
        compressed = base64.b64decode(compressed_str.encode('utf-8'))
        # Décompresser
        json_str = gzip.decompress(compressed).decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        # Si la décompression échoue, essayer de parser comme JSON normal (rétrocompatibilité)
        logger.warning(f"⚠️ Failed to decompress, trying as plain JSON: {e}")
        return json.loads(compressed_str)

# --- PHASE 3 : Cache de modèles Whisper ---
_model_cache = {}
_model_cache_lock = threading.Lock()
_MAX_CACHED_MODELS = 2  # Nombre maximum de modèles en cache (LRU)

# --- AJOUTS : Variables globales pour psutil ---
WORKER_PROCESS = None
WORKER_START_TIME = None

@worker_init.connect
def on_worker_init(**kwargs):
    """Initialise psutil quand le worker démarre."""
    global WORKER_PROCESS, WORKER_START_TIME
    try:
        WORKER_PROCESS = psutil.Process(os.getpid())
        WORKER_START_TIME = datetime.now()
        WORKER_PROCESS.cpu_percent(interval=None) # Initialiser la mesure
        logger.info(f"Worker {WORKER_PROCESS.pid} initialisé pour monitoring psutil.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de psutil: {e}")
# --- FIN AJOUTS ---


def get_api_client():
    """Charge le client API (une fois par worker) - Version refactorisée"""
    global _api_client
    if _api_client is None:
        logger.info(f"Initialisation du client API pour ce worker ({config.instance_name})...")
        _api_client = VocalyxAPIClientRefactored(config)
    return _api_client

def get_worker_service():
    """Charge le service worker (une fois par worker)"""
    api_client = get_api_client()
    return TranscriptionWorkerService(api_client)

def get_transcription_service(model_name: str = 'small'):
    """
    Charge le service de transcription avec cache par modèle (Phase 3 - Optimisation).
    
    Args:
        model_name: Nom du modèle Whisper (tiny, base, small, medium, large) ou chemin
        
    Returns:
        TranscriptionService: Service de transcription avec le modèle demandé
    """
    global _model_cache, _transcription_service
    
    # Normaliser le nom du modèle
    if not model_name:
        model_name = 'small'
    else:
        model_name = model_name.lower()
        # Si c'est un chemin, extraire le nom du modèle
        # Ex: "./models/openai-whisper-small" -> "small"
        if 'openai-whisper-' in model_name:
            model_name = model_name.split('openai-whisper-')[-1].split('/')[-1].split('\\')[-1]
        # Si c'est juste un chemin relatif, utiliser 'small' par défaut
        elif model_name.startswith('./') or model_name.startswith('/'):
            # Essayer d'extraire le nom du modèle du chemin
            parts = model_name.replace('\\', '/').split('/')
            for part in reversed(parts):
                if part in ['tiny', 'base', 'small', 'medium', 'large']:
                    model_name = part
                    break
            else:
                model_name = 'small'  # Fallback
    
    # Si aucun modèle spécifié, utiliser l'ancien comportement (rétrocompatibilité)
    if model_name == 'small' and _transcription_service is not None:
        # Vérifier si le service existant utilise le modèle par défaut
        if not hasattr(_transcription_service, 'model_name') or _transcription_service.model_name == 'small':
            return _transcription_service
    
    # Utiliser le cache de modèles
    with _model_cache_lock:
        # Vérifier si le modèle est déjà en cache
        if model_name in _model_cache:
            logger.info(f"✅ Using cached Whisper model: {model_name}")
            _model_cache[model_name]['last_used'] = time.time()
            return _model_cache[model_name]['service']
        
        # Si le cache est plein, supprimer le moins récemment utilisé (LRU)
        if len(_model_cache) >= _MAX_CACHED_MODELS:
            oldest_model = min(_model_cache.keys(), 
                             key=lambda k: _model_cache[k]['last_used'])
            logger.info(f"🗑️ Removing least recently used model from cache: {oldest_model}")
            del _model_cache[oldest_model]
        
        # Charger le nouveau modèle
        logger.info(f"🚀 Loading Whisper model into cache: {model_name} (cache: {len(_model_cache)}/{_MAX_CACHED_MODELS})")
        try:
            # Le TranscriptionService charge maintenant TOUS les modèles au démarrage
            # (Whisper + pyannote) avec des logs détaillés
            service = TranscriptionService(config, model_name=model_name)
            _model_cache[model_name] = {
                'service': service,
                'last_used': time.time()
            }
            logger.info(f"✅ Model {model_name} loaded and cached successfully")
            
            # Note: Le message "WORKER PRÊT" est maintenant affiché dans @worker_init
            # après le chargement de tous les modèles
            
            # Si c'est le modèle par défaut (small), mettre à jour aussi _transcription_service pour rétrocompatibilité
            if model_name == 'small':
                _transcription_service = service
            
            return service
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}", exc_info=True)
            raise

# --- FIN DES MODIFICATIONS GLOBALES ---


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
    
    # Vérifier si on doit utiliser le mode distribué
    if use_distributed is None:
        from pathlib import Path
        try:
            if file_path:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    import soundfile as sf
                    duration = sf.info(str(file_path_obj)).duration
                    # Utiliser le seuil depuis la config (par défaut 30s)
                    min_duration = 30  # TODO: Récupérer depuis config si possible
                    use_distributed = duration > min_duration
                    logger.info(
                        f"[{transcription_id}] 📊 DISTRIBUTION DECISION (worker) | "
                        f"Duration: {duration:.1f}s | "
                        f"Threshold: {min_duration}s | "
                        f"Mode: {'DISTRIBUTED' if use_distributed else 'CLASSIC'}"
                    )
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Could not determine distribution mode: {e}")
            use_distributed = False
    
    # Si mode distribué, déléguer à orchestrate_distributed_transcription
    if use_distributed and file_path:
        from pathlib import Path
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            logger.info(
                f"[{transcription_id}] 🚀 DISTRIBUTED MODE | "
                f"Delegating to orchestrate_distributed_transcription | "
                f"Worker: {config.instance_name}"
            )
            from celery import current_app as celery_current_app
            
            # Envoyer la tâche d'orchestration
            orchestrate_task = celery_current_app.send_task(
                'orchestrate_distributed_transcription',
                args=[transcription_id, str(file_path)],
                queue='transcription',
                countdown=1
            )
            
            logger.info(
                f"[{transcription_id}] ✅ DISTRIBUTED MODE | "
                f"Orchestration task enqueued: {orchestrate_task.id}"
            )
            
            return {
                "transcription_id": transcription_id,
                "task_id": self.request.id,
                "orchestration_task_id": orchestrate_task.id,
                "status": "queued_distributed",
                "mode": "distributed"
            }
    
    # MODE CLASSIQUE : Traitement direct
    logger.info(
        f"[{transcription_id}] 🎯 CLASSIC MODE STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"Mode: Single worker processing entire audio (non-distributed)"
    )
    start_time = time.time()
    
    try:
        use_vad = transcription.get('vad_enabled', True)
        use_diarization = transcription.get('diarization_enabled', False)
        whisper_model = transcription.get('whisper_model', 'small')  # Récupérer le modèle choisi
        
        logger.info(f"[{transcription_id}] 📁 File: {file_path} | VAD: {use_vad} | Diarization: {use_diarization} | Model: {whisper_model}")
        
        # 2. Mettre à jour le statut à "processing"
        api_client.update_transcription(transcription_id, {
            "status": "processing",
            "worker_id": config.instance_name
        })
        logger.info(f"[{transcription_id}] ⚙️ Status updated to 'processing'")
        
        # 3. Obtenir le service de transcription avec cache de modèles (Phase 3 - Optimisation)
        # Le cache réutilise les modèles déjà chargés, évitant 5-15s de chargement
        logger.info(f"[{transcription_id}] 🎤 Getting transcription service with model: {whisper_model} (cached)")
        transcription_service = get_transcription_service(model_name=whisper_model)
        
        # 4. Exécuter la transcription
        logger.info(f"[{transcription_id}] 🎤 Starting transcription with Whisper...")
        
        result = transcription_service.transcribe(
            file_path=file_path,
            use_vad=use_vad,
            use_diarization=use_diarization,
            transcription_id=transcription_id
        )
        
        logger.info(f"[{transcription_id}] ✅ Transcription service completed")
        
        processing_time = round(time.time() - start_time, 2)
        
        logger.info(
            f"[{transcription_id}] ✅ Transcription completed | "
            f"Duration: {result['duration']}s | "
            f"Processing: {processing_time}s | "
            f"Segments: {len(result['segments'])}"
        )
        
        # 4. Mettre à jour avec les résultats
        logger.info(f"[{transcription_id}] 💾 Saving results to API...")
        import json
        enrichment_requested = transcription.get('enrichment_requested', False)
        
        # Si l'enrichissement est demandé, mettre le statut à "transcribed" au lieu de "done"
        # Le statut sera changé à "done" uniquement quand l'enrichissement sera terminé
        status = "transcribed" if enrichment_requested else "done"
        
        api_client.update_transcription(transcription_id, {
            "status": status,
            "text": result["text"],
            "segments": json.dumps(result["segments"]),
            "language": result["language"],
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"])
        })
        
        logger.info(f"[{transcription_id}] 💾 Results saved to API (status: {status})")
        
        # 5. Si l'enrichissement est demandé, déclencher la tâche d'enrichissement
        if enrichment_requested:
            try:
                logger.info(f"[{transcription_id}] 🤖 Enrichment requested, triggering enrichment task...")
                # Importer la tâche d'enrichissement depuis celery_app
                from celery import current_app as celery_current_app
                enrich_task = celery_current_app.send_task(
                    'enrich_transcription',
                    args=[transcription_id],
                    queue='enrichment',  # Utiliser la queue d'enrichissement
                    countdown=1  # Démarrer après 1 seconde
                )
                logger.info(f"[{transcription_id}] ✅ Enrichment task enqueued: {enrich_task.id}")
            except Exception as enrich_error:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to enqueue enrichment task: {enrich_error}")
                # Ne pas échouer la transcription si l'enrichissement échoue à être enqueuée
                # On met simplement le statut d'enrichissement en erreur
                try:
                    api_client.update_transcription(transcription_id, {
                        "enrichment_status": "error",
                        "enrichment_error": f"Failed to enqueue enrichment: {str(enrich_error)}"
                    })
                except:
                    pass
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"]),
            "enrichment_triggered": enrichment_requested
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Error: {e}", exc_info=True)
        
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
        return {
            "status": "error",
            "transcription_id": transcription_id,
            "error": str(e)
        }

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
        
        # Mettre à jour le statut
        api_client.update_transcription(transcription_id, {
            "status": "processing",
            "worker_id": f"{config.instance_name}-orchestrator"
        })
        
        # 1. Pré-traiter l'audio
        logger.info(f"[{transcription_id}] 🔧 DISTRIBUTED ORCHESTRATION | Step 1/4: Preprocessing audio...")
        preprocessed = preprocess_audio(file_path_obj, preserve_stereo_for_diarization=transcription.get('diarization_enabled', False))
        processed_path_mono = preprocessed['mono']
        
        # 2. Découper en segments (forcer la découpe car on est en mode distribué)
        logger.info(
            f"[{transcription_id}] ✂️ DISTRIBUTED ORCHESTRATION | Step 2/4: Splitting audio into segments | "
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
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 2/4: Segmentation complete | "
            f"Segments created: {num_segments} | "
            f"Will be distributed across available workers"
        )
        
        if num_segments == 0:
            raise ValueError("No segments created")
        
        # 3. Stocker les métadonnées dans Redis
        redis_client = get_redis_client()
        segments_key = f"transcription:{transcription_id}:segments"
        # Timestamp de début pour calculer le temps réel écoulé
        orchestration_start_time = time.time()
        
        segments_metadata = {
            "transcription_id": transcription_id,
            "total_segments": num_segments,
            "completed_segments": 0,
            "segment_paths": [str(p) for p in segment_paths],
            "use_vad": use_vad,
            "use_diarization": transcription.get('diarization_enabled', False),
            "whisper_model": whisper_model,
            "processed_path_mono": str(processed_path_mono),
            "processed_path_stereo": str(preprocessed.get('stereo')) if preprocessed.get('stereo') else None,
            "is_stereo": preprocessed.get('is_stereo', False),
            "original_duration": get_audio_duration(file_path_obj),
            "orchestration_start_time": orchestration_start_time  # Timestamp de début pour calculer le temps réel
        }
        # Stocker avec compression si activée
        ttl = getattr(config, 'redis_transcription_ttl', 3600)
        segments_data = _compress_json(segments_metadata) if getattr(config, 'redis_transcription_compress', True) else json.dumps(segments_metadata)
        redis_client.setex(segments_key, ttl, segments_data)
        
        # Initialiser le compteur atomique à 0 (pour éviter les race conditions)
        counter_key = f"transcription:{transcription_id}:completed_count"
        # Utiliser set au lieu de setex pour s'assurer que c'est bien un entier
        redis_client.delete(counter_key)  # S'assurer qu'il n'existe pas déjà
        redis_client.set(counter_key, 0)
        redis_client.expire(counter_key, 3600)  # Expire après 1h
        
        # 4. Créer une tâche pour chaque segment
        logger.info(
            f"[{transcription_id}] 📤 DISTRIBUTED ORCHESTRATION | Step 3/4: Creating segment tasks | "
            f"Total segments: {num_segments} | "
            f"Queue: transcription | "
            f"Tasks will be distributed automatically by Celery"
        )
        segment_tasks = []
        from celery import current_app as celery_current_app
        
        for i, segment_path in enumerate(segment_paths):
            # Calculer l'offset temporel (start_time) pour ce segment
            # On va le calculer approximativement en fonction de l'index
            # Le worker de segment pourra ajuster avec la durée réelle
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
        
        # 5. Stocker les IDs des tâches
        tasks_key = f"transcription:{transcription_id}:segment_tasks"
        redis_client.setex(tasks_key, 3600, json.dumps(segment_tasks))
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 3/4: All segment tasks created | "
            f"Total tasks: {num_segments} | "
            f"Task IDs: {', '.join(segment_tasks[:5])}{'...' if len(segment_tasks) > 5 else ''} | "
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
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "status": "error",
                "error_message": f"Orchestration failed: {str(e)}"
            })
        except:
            pass
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
        redis_client = get_redis_client()
        segments_key = f"transcription:{transcription_id}:segments"
        metadata_json = redis_client.get(segments_key)
        
        if not metadata_json:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        # Décompresser si nécessaire
        try:
            metadata = _decompress_json(metadata_json)
        except:
            # Fallback : essayer comme JSON normal (rétrocompatibilité)
            metadata = json.loads(metadata_json)
        use_vad = metadata.get('use_vad', True)
        whisper_model = metadata.get('whisper_model', 'small')
        
        logger.info(
            f"[{transcription_id}] ⚙️ DISTRIBUTED SEGMENT | Worker {config.instance_name} processing | "
            f"Segment: {segment_index+1}/{total_segments} | "
            f"Model: {whisper_model} | VAD: {use_vad}"
        )
        
        # Transcrit le segment
        transcription_service = get_transcription_service(model_name=whisper_model)
        segment_path_obj = Path(segment_path)
        
        if not segment_path_obj.exists():
            raise FileNotFoundError(f"Segment not found: {segment_path}")
        
        text, segments_list, lang = transcription_service.transcribe_segment(
            segment_path_obj,
            use_vad=use_vad
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Calculer l'offset temporel pour ce segment
        # On utilise la durée cumulée des segments précédents
        time_offset = 0.0
        if segment_index > 0:
            # Récupérer les résultats des segments précédents pour calculer l'offset
            for prev_idx in range(segment_index):
                prev_result_key = f"transcription:{transcription_id}:segment:{prev_idx}:result"
                prev_result_json = redis_client.get(prev_result_key)
                if prev_result_json:
                    try:
                        # Décompresser si nécessaire (comme pour les métadonnées)
                        try:
                            prev_result = _decompress_json(prev_result_json)
                        except:
                            # Fallback : essayer comme JSON normal (rétrocompatibilité)
                            prev_result = json.loads(prev_result_json)
                        
                        if prev_result.get('segments'):
                            # Prendre le dernier timestamp du segment précédent
                            last_segment = prev_result['segments'][-1]
                            time_offset = last_segment.get('end', 0.0)
                            break  # On prend le dernier segment disponible
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        # Si le segment précédent n'est pas encore disponible ou invalide, ignorer
                        logger.debug(f"[{transcription_id}] Segment {prev_idx} not yet available or invalid, skipping offset calculation: {e}")
                        continue
        
        # Ajuster les timestamps avec l'offset
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
        
        result_key = f"transcription:{transcription_id}:segment:{segment_index}:result"
        # Stocker avec compression si activée
        ttl = getattr(config, 'redis_transcription_ttl', 3600)
        result_data = _compress_json(result) if getattr(config, 'redis_transcription_compress', True) else json.dumps(result)
        redis_client.setex(result_key, ttl, result_data)
        
        # Incrémenter le compteur de segments complétés de manière atomique (évite les race conditions)
        counter_key = f"transcription:{transcription_id}:completed_count"
        completed_count = int(redis_client.incr(counter_key))  # S'assurer que c'est un entier
        redis_client.expire(counter_key, 3600)  # Expire après 1h
        
        # Mettre à jour les métadonnées (sans le compteur, il est géré séparément)
        metadata['completed_segments'] = completed_count
        ttl = getattr(config, 'redis_transcription_ttl', 3600)
        metadata_data = _compress_json(metadata) if getattr(config, 'redis_transcription_compress', True) else json.dumps(metadata)
        redis_client.setex(segments_key, ttl, metadata_data)
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED SEGMENT COMPLETED | "
            f"Segment: {segment_index+1}/{total_segments} | "
            f"Worker: {config.instance_name} | "
            f"Processing time: {processing_time}s | "
            f"Progress: {completed_count}/{total_segments} segments done ({100*completed_count/total_segments:.1f}%)"
        )
        
        # Si tous les segments sont terminés, déclencher l'agrégation
        # Utiliser une opération atomique pour éviter les déclenchements multiples
        if completed_count >= total_segments:
            # Utiliser un verrou Redis pour s'assurer qu'un seul worker déclenche l'agrégation
            lock_key = f"transcription:{transcription_id}:aggregation_lock"
            lock_acquired = redis_client.set(lock_key, "1", ex=300, nx=True)  # nx=True = set only if not exists
            
            if lock_acquired:
                logger.info(
                    f"[{transcription_id}] 🎉 DISTRIBUTED MODE | All segments completed | "
                    f"Total: {total_segments} segments | "
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
    
    try:
        redis_client = get_redis_client()
        
        # Récupérer les métadonnées
        segments_key = f"transcription:{transcription_id}:segments"
        metadata_json = redis_client.get(segments_key)
        
        if not metadata_json:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        # Décompresser si nécessaire
        try:
            metadata = _decompress_json(metadata_json)
        except:
            # Fallback : essayer comme JSON normal (rétrocompatibilité)
            metadata = json.loads(metadata_json)
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
        max_segment_time = 0.0  # Temps maximum d'un segment (car traitement en parallèle)
        segment_processing_times = []  # Pour statistiques
        
        for i in range(total_segments):
            result_key = f"transcription:{transcription_id}:segment:{i}:result"
            result_json = redis_client.get(result_key)
            
            if not result_json:
                raise ValueError(f"Result not found for segment {i} of transcription {transcription_id}")
            
            # Décompresser si nécessaire
            try:
                result = _decompress_json(result_json)
            except:
                # Fallback : essayer comme JSON normal (rétrocompatibilité)
                result = json.loads(result_json)
            all_segments.extend(result['segments'])
            full_text += result['text'] + " "
            segment_time = result.get('processing_time', 0.0)
            segment_processing_times.append(segment_time)
            max_segment_time = max(max_segment_time, segment_time)  # Temps max car traitement en parallèle
            
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
        
        # Trier les segments par timestamp (au cas où ils arrivent dans le désordre)
        all_segments.sort(key=lambda x: x['start'])
        logger.info(
            f"[{transcription_id}] 🔄 DISTRIBUTED AGGREGATION | Step 2/3: Segments sorted by timestamp | "
            f"Total segments: {len(all_segments)}"
        )
        
        # Diarisation si demandée
        if use_diarization:
            logger.info(f"[{transcription_id}] 🎤 Running speaker diarization on aggregated segments...")
            try:
                transcription_service = get_transcription_service(model_name=metadata.get('whisper_model', 'small'))
                if transcription_service.diarization_service and transcription_service.diarization_service.pipeline:
                    diarization_audio_path = Path(metadata['processed_path_stereo']) if metadata.get('processed_path_stereo') else Path(metadata['processed_path_mono'])
                    
                    if diarization_audio_path.exists():
                        diarization_segments = transcription_service.diarization_service.diarize(diarization_audio_path)
                        
                        if diarization_segments:
                            all_segments = transcription_service.diarization_service.assign_speakers_to_segments(
                                all_segments,
                                diarization_segments
                            )
                            logger.info(f"[{transcription_id}] ✅ Diarization completed and assigned to segments")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Diarization error: {e}")
        
        # Sauvegarder le résultat final
        api_client = get_api_client()
        transcription = api_client.get_transcription(transcription_id)
        enrichment_requested = transcription.get('enrichment_requested', False) if transcription else False
        
        status = "transcribed" if enrichment_requested else "done"
        aggregation_time = round(time.time() - start_time, 2)
        
        # Le temps réel de traitement est le temps écoulé depuis le début de l'orchestration
        # (ou le max des segments + agrégation si pas de timestamp de début)
        if orchestration_start_time:
            total_processing_time = round(time.time() - orchestration_start_time, 2)
        else:
            # Fallback : max segment + agrégation
            total_processing_time = round(max_segment_time + aggregation_time, 2)
        
        api_client.update_transcription(transcription_id, {
            "status": status,
            "text": full_text.strip(),
            "segments": json.dumps(all_segments),
            "language": language_detected,
            "duration": metadata.get('original_duration', 0.0),
            "processing_time": total_processing_time,
            "segments_count": len(all_segments)
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
        
        # Nettoyer les données Redis (utiliser pipeline pour performance)
        try:
            # Utiliser un pipeline Redis pour supprimer toutes les clés en une seule opération
            pipe = redis_client.pipeline()
            for i in range(total_segments):
                pipe.delete(f"transcription:{transcription_id}:segment:{i}:result")
            pipe.delete(segments_key)
            pipe.delete(f"transcription:{transcription_id}:segment_tasks")
            pipe.delete(f"transcription:{transcription_id}:completed_count")
            pipe.delete(f"transcription:{transcription_id}:aggregation_lock")
            pipe.execute()  # Exécuter toutes les suppressions en une seule fois
            logger.debug(f"[{transcription_id}] 🧹 Redis cleanup completed (pipeline)")
            
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
            try:
                from celery import current_app as celery_current_app
                enrich_task = celery_current_app.send_task(
                    'enrich_transcription',
                    args=[transcription_id],
                    queue='enrichment',
                    countdown=1
                )
                logger.info(f"[{transcription_id}] ✅ Enrichment task enqueued: {enrich_task.id}")
            except Exception as enrich_error:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to enqueue enrichment task: {enrich_error}")
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "segments_count": len(all_segments),
            "total_processing_time": total_processing_time
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Aggregation error: {e}", exc_info=True)
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "status": "error",
                "error_message": f"Aggregation failed: {str(e)}"
            })
        except:
            pass
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