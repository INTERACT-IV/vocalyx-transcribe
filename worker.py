"""
vocalyx-transcribe/worker.py
Worker Celery pour la transcription audio
"""

import logging
import time
import os
import psutil
import threading
from datetime import datetime
from celery.signals import worker_init
from celery.worker.control import Panel

from celery import Celery
from config import Config
from transcription_service import TranscriptionService
from api_client import VocalyxAPIClient

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

# --- PHASE 3 : Cache de modèles Whisper ---
_model_cache = {}
_model_cache_lock = threading.Lock()
_MAX_CACHED_MODELS = 2  # Limiter à 2 modèles en cache pour économiser la mémoire

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
    """Charge le client API (une fois par worker)"""
    global _api_client
    if _api_client is None:
        logger.info(f"Initialisation du client API pour ce worker ({config.instance_name})...")
        _api_client = VocalyxAPIClient(config)
    return _api_client

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
            service = TranscriptionService(config, model_name=model_name)
            _model_cache[model_name] = {
                'service': service,
                'last_used': time.time()
            }
            logger.info(f"✅ Model {model_name} loaded and cached successfully")
            
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
    name='transcribe_audio',
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=1800,
    time_limit=2100,
    acks_late=True,
    reject_on_worker_lost=True
)
def transcribe_audio_task(self, transcription_id: str):
    """
    Tâche de transcription exécutée par le worker.
    """
    
    logger.info(f"[{transcription_id}] 🎯 Task started by worker {config.instance_name}")
    start_time = time.time()
    
    try:
        # Assure que le client API est initialisé
        api_client = get_api_client()

        # 1. Récupérer les informations de la transcription depuis l'API
        logger.info(f"[{transcription_id}] 📡 Fetching transcription data from API...")
        transcription = api_client.get_transcription(transcription_id)
        
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        file_path = transcription.get('file_path')
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
        api_client.update_transcription(transcription_id, {
            "status": "done",
            "text": result["text"],
            "segments": json.dumps(result["segments"]),
            "language": result["language"],
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"])
        })
        
        logger.info(f"[{transcription_id}] 💾 Results saved to API")
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "duration": result["duration"],
            "processing_time": processing_time,
            "segments_count": len(result["segments"])
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

if __name__ == "__main__":
    # ... (Le reste de votre fichier __main__ est parfait et n'a pas besoin d'être modifié) ...
    logger.info(f"🚀 Starting Celery worker: {config.instance_name}")
    celery_app.worker_main([
        'worker',
        f'--loglevel={config.log_level.lower()}',
        f'--concurrency={config.max_workers}',
        f'--hostname={config.instance_name}@%h'
    ])