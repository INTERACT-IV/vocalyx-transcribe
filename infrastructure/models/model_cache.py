"""
Cache de modèles Whisper avec stratégie LRU
"""

import logging
import time
import threading
from typing import Optional, Dict
from transcription_service import TranscriptionService

logger = logging.getLogger("vocalyx")


class ModelCache:
    """Cache LRU pour les modèles Whisper"""
    
    def __init__(self, max_models: int = 2):
        self.max_models = max_models
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def get(self, model_name: str, config) -> Optional[TranscriptionService]:
        """
        Récupère un modèle depuis le cache ou le charge si nécessaire.
        
        Args:
            model_name: Nom du modèle (normalisé)
            config: Configuration du worker
            
        Returns:
            TranscriptionService: Service de transcription avec le modèle
        """
        normalized_name = self._normalize_model_name(model_name)
        
        with self._lock:
            if normalized_name in self._cache:
                logger.info(f"✅ Using cached Whisper model: {normalized_name}")
                self._cache[normalized_name]['last_used'] = time.time()
                return self._cache[normalized_name]['service']
            
            # Si le cache est plein, supprimer le moins récemment utilisé
            if len(self._cache) >= self.max_models:
                self._evict_lru()
            
            # Charger le nouveau modèle
            logger.info(f"🚀 Loading Whisper model into cache: {normalized_name} (cache: {len(self._cache)}/{self.max_models})")
            try:
                service = TranscriptionService(config, model_name=normalized_name)
                self._cache[normalized_name] = {
                    'service': service,
                    'last_used': time.time()
                }
                logger.info(f"✅ Model {normalized_name} loaded and cached successfully")
                return service
            except Exception as e:
                logger.error(f"❌ Failed to load model {normalized_name}: {e}", exc_info=True)
                raise
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalise le nom du modèle"""
        if not model_name:
            return 'small'
        
        model_name = model_name.lower()
        
        # Si c'est un chemin, extraire le nom du modèle
        if 'openai-whisper-' in model_name:
            model_name = model_name.split('openai-whisper-')[-1].split('/')[-1].split('\\')[-1]
        elif model_name.startswith('./') or model_name.startswith('/'):
            parts = model_name.replace('\\', '/').split('/')
            for part in reversed(parts):
                if part in ['tiny', 'base', 'small', 'medium', 'large']:
                    return part
            return 'small'
        
        return model_name if model_name in ['tiny', 'base', 'small', 'medium', 'large'] else 'small'
    
    def _evict_lru(self):
        """Supprime le modèle le moins récemment utilisé"""
        if not self._cache:
            return
        
        oldest_model = min(self._cache.keys(), key=lambda k: self._cache[k]['last_used'])
        logger.info(f"🗑️ Removing least recently used model from cache: {oldest_model}")
        del self._cache[oldest_model]

