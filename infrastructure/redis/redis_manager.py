"""
Gestionnaire Redis pour les opérations de transcription distribuée
"""

import json
import gzip
import base64
import logging
import redis
from typing import Dict, Optional

logger = logging.getLogger("vocalyx")


class RedisCompressionManager:
    """Gère la compression/décompression des données JSON pour Redis"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def compress(self, data: dict) -> str:
        """Compresse un dictionnaire JSON pour économiser la mémoire Redis"""
        if not self.enabled:
            return json.dumps(data)
        
        json_str = json.dumps(data)
        compressed = gzip.compress(json_str.encode('utf-8'), compresslevel=6)
        return base64.b64encode(compressed).decode('utf-8')
    
    def decompress(self, compressed_str: str) -> dict:
        """Décompresse une chaîne JSON compressée"""
        if not self.enabled:
            return json.loads(compressed_str)
        
        try:
            compressed = base64.b64decode(compressed_str.encode('utf-8'))
            json_str = gzip.decompress(compressed).decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"⚠️ Failed to decompress, trying as plain JSON: {e}")
            return json.loads(compressed_str)


class RedisTranscriptionManager:
    """Gère les opérations Redis pour la transcription distribuée"""
    
    def __init__(self, redis_client: redis.Redis, compression_manager: RedisCompressionManager):
        self.client = redis_client
        self.compression = compression_manager
    
    def store_metadata(self, transcription_id: str, metadata: dict, ttl: int = 3600):
        """Stocke les métadonnées de transcription"""
        key = f"transcription:{transcription_id}:segments"
        data = self.compression.compress(metadata) if self.compression.enabled else json.dumps(metadata)
        self.client.setex(key, ttl, data)
    
    def get_metadata(self, transcription_id: str) -> Optional[dict]:
        """Récupère les métadonnées de transcription"""
        key = f"transcription:{transcription_id}:segments"
        data = self.client.get(key)
        if not data:
            return None
        
        try:
            return self.compression.decompress(data)
        except:
            return json.loads(data)
    
    def store_segment_result(self, transcription_id: str, segment_index: int, result: dict, ttl: int = 3600):
        """Stocke le résultat d'un segment"""
        key = f"transcription:{transcription_id}:segment:{segment_index}:result"
        data = self.compression.compress(result) if self.compression.enabled else json.dumps(result)
        self.client.setex(key, ttl, data)
    
    def get_segment_result(self, transcription_id: str, segment_index: int) -> Optional[dict]:
        """Récupère le résultat d'un segment"""
        key = f"transcription:{transcription_id}:segment:{segment_index}:result"
        data = self.client.get(key)
        if not data:
            return None
        
        try:
            return self.compression.decompress(data)
        except:
            return json.loads(data)
    
    def increment_completed_count(self, transcription_id: str) -> int:
        """Incrémente atomiquement le compteur de segments complétés"""
        key = f"transcription:{transcription_id}:completed_count"
        count = int(self.client.incr(key))
        # Le TTL sera renouvelé par refresh_ttl si nécessaire
        self.client.expire(key, 3600)
        return count
    
    def refresh_ttl(self, transcription_id: str, ttl: int, total_segments: int = None):
        """
        Renouvelle le TTL de toutes les clés Redis associées à une transcription.
        Utile pour éviter l'expiration pendant un traitement long.
        
        Args:
            transcription_id: ID de la transcription
            ttl: Nouveau TTL en secondes
            total_segments: Nombre total de segments (optionnel, pour renouveler les résultats de segments)
        """
        pipe = self.client.pipeline()
        
        # Renouveler les métadonnées
        pipe.expire(f"transcription:{transcription_id}:segments", ttl)
        pipe.expire(f"transcription:{transcription_id}:completed_count", ttl)
        pipe.expire(f"transcription:{transcription_id}:aggregation_lock", ttl)
        pipe.expire(f"transcription:{transcription_id}:segment_tasks", ttl)
        pipe.expire(f"transcription:{transcription_id}:diarization_completed_count", ttl)
        pipe.expire(f"transcription:{transcription_id}:diarization_tasks", ttl)
        
        # Renouveler les résultats de segments si le nombre total est fourni
        if total_segments is not None:
            for i in range(total_segments):
                pipe.expire(f"transcription:{transcription_id}:segment:{i}:result", ttl)
                pipe.expire(f"transcription:{transcription_id}:diarization:{i}:result", ttl)
        
        pipe.execute()
    
    def reset_completed_count(self, transcription_id: str, ttl: int = 3600):
        """Réinitialise le compteur de segments complétés"""
        key = f"transcription:{transcription_id}:completed_count"
        self.client.delete(key)
        self.client.set(key, 0)
        self.client.expire(key, ttl)
    
    def acquire_aggregation_lock(self, transcription_id: str, timeout: int = 300) -> bool:
        """Tente d'acquérir un verrou pour l'agrégation (évite les déclenchements multiples)"""
        key = f"transcription:{transcription_id}:aggregation_lock"
        return bool(self.client.set(key, "1", ex=timeout, nx=True))
    
    def store_diarization_result(self, transcription_id: str, segment_index: int, result: dict, ttl: int = 3600):
        """Stocke le résultat de diarisation d'un segment"""
        key = f"transcription:{transcription_id}:diarization:{segment_index}:result"
        data = self.compression.compress(result) if self.compression.enabled else json.dumps(result)
        self.client.setex(key, ttl, data)
    
    def get_diarization_result(self, transcription_id: str, segment_index: int) -> Optional[dict]:
        """Récupère le résultat de diarisation d'un segment"""
        key = f"transcription:{transcription_id}:diarization:{segment_index}:result"
        data = self.client.get(key)
        if not data:
            return None
        
        try:
            return self.compression.decompress(data)
        except:
            return json.loads(data)
    
    def increment_diarization_count(self, transcription_id: str) -> int:
        """Incrémente atomiquement le compteur de segments de diarisation complétés"""
        key = f"transcription:{transcription_id}:diarization_completed_count"
        count = int(self.client.incr(key))
        # Le TTL sera renouvelé par refresh_ttl si nécessaire
        self.client.expire(key, 3600)
        return count
    
    def check_ttl_health(self, transcription_id: str, threshold_percent: float = 0.2) -> dict:
        """
        Vérifie la santé du TTL pour une transcription.
        
        Args:
            transcription_id: ID de la transcription
            threshold_percent: Seuil d'alerte (ex: 0.2 = 20% du TTL restant)
        
        Returns:
            dict: {
                'healthy': bool,
                'ttl_remaining': int,  # Secondes restantes
                'ttl_original': int,   # TTL original
                'percent_remaining': float,
                'alert': bool,
                'message': str
            }
        """
        key = f"transcription:{transcription_id}:segments"
        ttl_remaining = self.client.ttl(key)
        
        if ttl_remaining == -2:  # Clé n'existe plus
            return {
                'healthy': False,
                'ttl_remaining': 0,
                'ttl_original': 0,
                'percent_remaining': 0.0,
                'alert': True,
                'message': f'Key expired or not found for transcription {transcription_id}'
            }
        
        if ttl_remaining == -1:  # Pas de TTL (ne devrait pas arriver)
            return {
                'healthy': True,
                'ttl_remaining': -1,
                'ttl_original': -1,
                'percent_remaining': 100.0,
                'alert': False,
                'message': 'No TTL set (unlimited)'
            }
        
        # Récupérer le TTL original depuis les métadonnées
        metadata = self.get_metadata(transcription_id)
        ttl_original = metadata.get('ttl_original', ttl_remaining) if metadata else ttl_remaining
        
        percent_remaining = (ttl_remaining / ttl_original * 100) if ttl_original > 0 else 0
        alert = percent_remaining < (threshold_percent * 100)
        
        return {
            'healthy': not alert and ttl_remaining > 0,
            'ttl_remaining': ttl_remaining,
            'ttl_original': ttl_original,
            'percent_remaining': percent_remaining,
            'alert': alert,
            'message': f'TTL: {ttl_remaining}s remaining ({percent_remaining:.1f}%)'
        }
    
    def cleanup(self, transcription_id: str, total_segments: int):
        """Nettoie toutes les clés Redis associées à une transcription"""
        pipe = self.client.pipeline()
        for i in range(total_segments):
            pipe.delete(f"transcription:{transcription_id}:segment:{i}:result")
        pipe.delete(f"transcription:{transcription_id}:segments")
        pipe.delete(f"transcription:{transcription_id}:segment_tasks")
        pipe.delete(f"transcription:{transcription_id}:completed_count")
        pipe.delete(f"transcription:{transcription_id}:aggregation_lock")
        pipe.execute()
        logger.debug(f"[{transcription_id}] 🧹 Redis cleanup completed")

