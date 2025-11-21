"""
VocalyxAPIClient - Client HTTP refactorisé pour communiquer avec vocalyx-api
"""

import logging
from typing import Dict, Optional
import httpx

logger = logging.getLogger("vocalyx")


class VocalyxAPIClient:
    """
    Client HTTP pour communiquer avec vocalyx-api.
    Le worker utilise ce client pour récupérer et mettre à jour les transcriptions.
    """
    
    def __init__(self, config):
        self.base_url = config.api_url.rstrip('/')
        self.internal_key = config.internal_api_key
        self.timeout = httpx.Timeout(60.0, connect=5.0)
        
        # Client synchrone (suffisant pour le worker)
        self.client = httpx.Client(timeout=self.timeout)
        
        logger.info(f"API Client initialized: {self.base_url}")
        
        # Vérifier la connexion à l'API au démarrage
        self._verify_connection()
    
    def _verify_connection(self):
        """Vérifie la connexion à l'API au démarrage"""
        try:
            response = self.client.get(
                f"{self.base_url}/health",
                timeout=httpx.Timeout(5.0)
            )
            response.raise_for_status()
            health = response.json()
            
            if health.get("status") == "healthy":
                logger.info("✅ API connection verified")
            else:
                logger.warning(f"⚠️ API health check returned: {health}")
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            logger.error("⚠️ Worker will start but may fail to process tasks")
    
    def _get_headers(self) -> Dict[str, str]:
        """Génère les headers d'authentification interne"""
        return {
            "X-Internal-Key": self.internal_key
        }
    
    def get_transcription(self, transcription_id: str) -> Optional[Dict]:
        """
        Récupère une transcription par son ID.
        
        Args:
            transcription_id: ID de la transcription
            
        Returns:
            dict: Données de la transcription ou None si non trouvée
        """
        try:
            response = self.client.get(
                f"{self.base_url}/api/transcriptions/{transcription_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Transcription {transcription_id} not found")
                return None
            logger.error(f"HTTP error getting transcription: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Error getting transcription: {e}")
            raise
    
    def update_transcription(self, transcription_id: str, data: Dict) -> Dict:
        """
        Met à jour une transcription.
        
        Args:
            transcription_id: ID de la transcription
            data: Données à mettre à jour
            
        Returns:
            dict: Transcription mise à jour
        """
        try:
            response = self.client.patch(
                f"{self.base_url}/api/transcriptions/{transcription_id}",
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error updating transcription: {e}")
            raise
    
    def close(self):
        """Ferme le client HTTP"""
        self.client.close()

