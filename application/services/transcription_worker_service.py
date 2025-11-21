"""
TranscriptionWorkerService - Service pour la gestion des tâches de transcription
"""

import logging
import json
import time
from typing import Dict, Optional
from infrastructure.api.api_client import VocalyxAPIClient

logger = logging.getLogger("vocalyx")


class TranscriptionWorkerService:
    """Service pour gérer les tâches de transcription du worker Celery"""
    
    def __init__(self, api_client: VocalyxAPIClient):
        self.api_client = api_client
    
    def get_transcription(self, transcription_id: str) -> Optional[Dict]:
        """Récupère une transcription depuis l'API"""
        try:
            return self.api_client.get_transcription(transcription_id)
        except Exception as e:
            logger.error(f"[{transcription_id}] Error getting transcription: {e}")
            return None
    
    def mark_as_processing(self, transcription_id: str, worker_id: str) -> bool:
        """Marque une transcription comme en cours de traitement"""
        try:
            self.api_client.update_transcription(transcription_id, {
                "status": "processing",
                "worker_id": worker_id
            })
            logger.info(f"[{transcription_id}] Status updated to 'processing'")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error updating status to processing: {e}")
            return False
    
    def mark_as_done(
        self,
        transcription_id: str,
        text: str,
        segments: list,
        language: str,
        duration: float,
        processing_time: float
    ) -> bool:
        """Marque une transcription comme terminée avec ses résultats"""
        try:
            self.api_client.update_transcription(transcription_id, {
                "status": "done",
                "text": text,
                "segments": json.dumps(segments),
                "language": language,
                "duration": duration,
                "processing_time": processing_time,
                "segments_count": len(segments)
            })
            logger.info(f"[{transcription_id}] Results saved to API")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error saving results: {e}")
            return False
    
    def mark_as_error(self, transcription_id: str, error_message: str) -> bool:
        """Marque une transcription comme échouée"""
        try:
            self.api_client.update_transcription(transcription_id, {
                "status": "error",
                "error_message": str(error_message)
            })
            logger.error(f"[{transcription_id}] Marked as error: {error_message}")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error marking as error: {e}")
            return False

