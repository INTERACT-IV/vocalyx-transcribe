"""
vocalyx-transcribe/worker.py
Worker Celery pour la transcription audio
"""

import logging
import time
import os
import psutil
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

def get_transcription_service():
    """Charge le service de transcription (une fois par worker)"""
    global _transcription_service
    if _transcription_service is None:
        logger.info(f"Initialisation du TranscriptionService pour ce worker ({config.instance_name})...")
        _transcription_service = TranscriptionService(config)
        logger.info("Service de transcription (et modèle Whisper) chargé.")
    return _transcription_service

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
        # Assure que les services sont initialisés (nécessaire pour le handler ci-dessus)
        api_client = get_api_client()
        transcription_service = get_transcription_service()

        # 1. Récupérer les informations de la transcription depuis l'API
        logger.info(f"[{transcription_id}] 📡 Fetching transcription data from API...")
        transcription = api_client.get_transcription(transcription_id)
        
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        file_path = transcription.get('file_path')
        use_vad = transcription.get('vad_enabled', True)
        
        logger.info(f"[{transcription_id}] 📁 File: {file_path} | VAD: {use_vad}")
        
        # 2. Mettre à jour le statut à "processing"
        api_client.update_transcription(transcription_id, {
            "status": "processing",
            "worker_id": config.instance_name
        })
        logger.info(f"[{transcription_id}] ⚙️ Status updated to 'processing'")
        
        # 3. Exécuter la transcription
        logger.info(f"[{transcription_id}] 🎤 Starting transcription with Whisper...")
        
        result = transcription_service.transcribe(
            file_path=file_path,
            use_vad=use_vad
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
        
        # Mettre à jour les statistiques du service
        transcription_service.update_stats(
            audio_duration=result["duration"],
            processing_time=processing_time
        )
        
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