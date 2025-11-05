"""
vocalyx-transcribe/worker.py
Worker Celery pour la transcription audio
"""

import logging
import time
from celery import Celery
from config import Config
from transcription_service import TranscriptionService
from api_client import VocalyxAPIClient

# Initialiser la configuration
config = Config()

# Configurer le logging
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

# Initialiser le client API
api_client = VocalyxAPIClient(config)

# Initialiser le service de transcription
transcription_service = TranscriptionService(config)

# Créer l'application Celery (connexion au même broker que vocalyx-api)
celery_app = Celery(
    'vocalyx-transcribe',
    broker=config.celery_broker_url,
    backend=config.celery_result_backend
)

# Configuration de Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Performance
    worker_prefetch_multiplier=1,  # Prendre 1 tâche à la fois (équitable entre workers)
    worker_max_tasks_per_child=10,  # Redémarrer après 10 tâches (libère RAM/VRAM)
    
    # Retry
    task_acks_late=True,  # Acquitter la tâche seulement après succès
    task_reject_on_worker_lost=True,  # Re-enqueue si le worker crash

    # Connexion
    broker_connection_retry_on_startup=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

@celery_app.task(
    bind=True,
    name='transcribe_audio',
    max_retries=3,
    default_retry_delay=60
)
def transcribe_audio_task(self, transcription_id: str):
    """
    Tâche de transcription exécutée par le worker.
    
    Cette tâche est définie dans vocalyx-api mais EXÉCUTÉE ici.
    
    Args:
        self: Instance de la tâche (bind=True)
        transcription_id: ID de la transcription à traiter
        
    Returns:
        dict: Résultat de la transcription
    """
    
    logger.info(f"[{transcription_id}] 🎯 Task started by worker {config.instance_name}")
    start_time = time.time()
    
    try:
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
        
        processing_time = round(time.time() - start_time, 2)
        
        logger.info(
            f"[{transcription_id}] ✅ Transcription completed | "
            f"Duration: {result['duration']}s | "
            f"Processing: {processing_time}s | "
            f"Segments: {len(result['segments'])}"
        )
        
        # 4. Mettre à jour avec les résultats
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
            api_client.update_transcription(transcription_id, {
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
    """
    Lancement du worker depuis la ligne de commande.
    
    Usage:
        python worker.py
        
    Ou avec Celery directement:
        celery -A worker.celery_app worker --loglevel=info --concurrency=2
    """
    logger.info(f"🚀 Starting Celery worker: {config.instance_name}")
    logger.info(f"📊 Model: {config.model} | Device: {config.device}")
    logger.info(f"⚙️ Max Workers: {config.max_workers} | VAD: {config.vad_enabled}")
    logger.info(f"🔗 Broker: {config.celery_broker_url}")
    logger.info(f"📡 API: {config.api_url}")
    
    # Lancer le worker
    celery_app.worker_main([
        'worker',
        f'--loglevel={config.log_level.lower()}',
        f'--concurrency={config.max_workers}',
        f'--hostname={config.instance_name}@%h'
    ])