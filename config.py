"""
vocalyx-transcribe/config.py
Configuration du worker (adapté pour l'architecture microservices)
"""

import os
import logging
import configparser
from pathlib import Path
from typing import Set
try:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
except (ImportError, NotImplementedError):
    CPU_COUNT = os.cpu_count() or 4  # Fallback à 4 si détection impossible

logger = logging.getLogger("vocalyx")

class Config:
    """Charge et gère la configuration depuis config.ini"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        
        if not os.path.exists(config_file):
            self._create_default_config()
        
        self.config.read(config_file)
        self._load_settings()
        
    def _create_default_config(self):
        """Crée un fichier de configuration par défaut"""
        config = configparser.ConfigParser()
        
        config['CORE'] = {
            'instance_name': 'worker-01'
        }
        
        config['API'] = {
            'url': 'http://localhost:8000',
            'timeout': '60'
        }
        
        config['CELERY'] = {
            'broker_url': 'redis://localhost:6379/0',
            'result_backend': 'redis://localhost:6379/0'
        }
        
        config['REDIS_TRANSCRIPTION'] = {
            # DB Redis dédiée pour les opérations de transcription (isolation des données)
            'url': 'redis://localhost:6379/2',
            # Compression des données JSON (réduit la mémoire et le réseau)
            'compress_data': 'true',
            # TTL par défaut pour les segments (en secondes)
            'default_ttl': '3600'
        }
        
        config['WHISPER'] = {
            'model': './models/transcribe/openai-whisper-small',
            'device': 'cpu',
            'compute_type': 'int8',
            'cpu_threads': '8',
            'language': 'fr'
        }
        
        config['PERFORMANCE'] = {
            'max_workers': '2',
            'segment_length_ms': '45000',
            'vad_enabled': 'true',
            'beam_size': '1',  # Optimisé pour CPU (greedy search, plus rapide)
            'best_of': '1',    # Pas de recherche multiple (optimisation CPU)
            'temperature': '0.0'
        }
        
        config['PATHS'] = {
            'upload_dir': './shared_uploads'
        }
        
        config['VAD'] = {
            'enabled': 'true',
            'min_silence_len': '500',
            'silence_thresh': '-40',
            'vad_threshold': '0.5',
            'min_speech_duration_ms': '250',
            'min_silence_duration_ms': '500'
        }
        
        config['DIARIZATION'] = {
            'enabled': 'false',
            'model': 'pyannote/speaker-diarization-3.1',
            'model_path': '/app/models/transcribe/pyannote-speaker-diarization',
            'hf_token': '',
            # Paramètres d'optimisation de performance
            'min_speakers': '',  # Nombre minimum de locuteurs (vide = auto)
            'max_speakers': '',  # Nombre maximum de locuteurs (vide = auto)
            'chunk_duration_s': '300',  # Traitement par chunks (0 = désactivé, 300 = 5min recommandé pour réduire CPU)
            'chunk_overlap_s': '5',  # Chevauchement entre chunks (secondes)
            'segmentation_threshold': '0.5',  # Seuil de segmentation (0.0-1.0)
            'clustering_threshold': '0.7',  # Seuil de clustering (0.0-1.0)
            'use_gpu_if_available': 'true',  # Utiliser GPU si disponible
            # Optimisations CPU
            'num_threads': '0',  # Nombre de threads PyTorch (0 = tous, 2-4 = recommandé pour réduire CPU)
            'reduce_resolution': 'false',  # Réduire résolution audio (true = 8kHz, plus rapide)
            'fast_mode': 'false'  # Mode économique (moins de CPU, légèrement moins précis)
        }
        
        config['LOGGING'] = {
            'level': 'INFO',
            'file_enabled': 'true',
            'file_path': 'logs/vocalyx-transcribe.log',
            'colored': 'true'
        }
        
        config['SECURITY'] = {
            'internal_api_key': 'CHANGE_ME_SECRET_INTERNAL_KEY_12345'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        logging.info(f"✅ Created default config file: {self.config_file}")
    
    def _load_settings(self):
        """Charge les paramètres dans des attributs"""
        
        # CORE
        self.instance_name = os.environ.get(
            'INSTANCE_NAME', 
            self.config.get('CORE', 'instance_name', fallback=f"worker-{os.getpid()}")
        )
        
        # API
        self.api_url = os.environ.get(
            'VOCALYX_API_URL', 
            self.config.get('API', 'url')
        )
        self.api_timeout = self.config.getint('API', 'timeout', fallback=60)
        
        # CELERY
        self.celery_broker_url = os.environ.get(
            'CELERY_BROKER_URL', 
            self.config.get('CELERY', 'broker_url')
        )
        self.celery_result_backend = os.environ.get(
            'CELERY_RESULT_BACKEND', 
            self.config.get('CELERY', 'result_backend')
        )
        
        # REDIS TRANSCRIPTION (DB dédiée pour les segments)
        # Utilise DB 2 par défaut pour isoler des opérations Celery (DB 0) et autres (DB 1)
        # PRIORITÉ: Variable d'environnement > config.ini > fallback depuis CELERY_BROKER_URL
        redis_transcription_url = os.environ.get('REDIS_TRANSCRIPTION_URL', None)
        source = "environment variable"
        
        if not redis_transcription_url:
            # Essayer depuis config.ini seulement si la section existe
            try:
                redis_transcription_url = self.config.get('REDIS_TRANSCRIPTION', 'url')
                source = "config.ini"
            except (configparser.NoSectionError, configparser.NoOptionError):
                redis_transcription_url = None
        
        if redis_transcription_url:
            self.redis_transcription_url = redis_transcription_url
            logger.info(f"✅ Redis transcription URL ({source}): {redis_transcription_url}")
        else:
            # Fallback : utiliser DB 2 de la même instance Redis (depuis CELERY_BROKER_URL)
            base_redis_url = self.celery_broker_url.rsplit('/', 1)[0]  # Enlever /0
            self.redis_transcription_url = f"{base_redis_url}/2"
            logger.info(f"✅ Redis transcription URL (fallback from CELERY_BROKER_URL): {self.redis_transcription_url}")
        
        self.redis_transcription_compress = self.config.getboolean(
            'REDIS_TRANSCRIPTION', 'compress_data', fallback=True
        )
        self.redis_transcription_ttl = self.config.getint(
            'REDIS_TRANSCRIPTION', 'default_ttl', fallback=3600
        )
        
        # WHISPER
        self.model = os.environ.get(
            'WHISPER_MODEL', 
            self.config.get('WHISPER', 'model')
        )
        self.device = os.environ.get(
            'WHISPER_DEVICE', 
            self.config.get('WHISPER', 'device')
        )
        self.compute_type = os.environ.get(
            'WHISPER_COMPUTE_TYPE', 
            self.config.get('WHISPER', 'compute_type')
        )
        self.cpu_threads = self.config.getint('WHISPER', 'cpu_threads')
        self.language = os.environ.get(
            'WHISPER_LANGUAGE', 
            self.config.get('WHISPER', 'language')
        )
        
        # PERFORMANCE
        # Détection automatique du nombre de cores CPU
        self.cpu_count = CPU_COUNT
        logger.info(f"🔍 Detected CPU: {self.cpu_count} core(s)")
        
        self.max_workers = os.environ.get(
            'MAX_WORKERS', 
            self.config.getint('PERFORMANCE', 'max_workers')
        )
        
        # Découpage adaptatif : taille des segments selon le CPU
        # CPU faible (< 4 cores) : 25s (segments plus courts = moins de mémoire)
        # CPU moyen (4-8 cores) : 35s (équilibre)
        # CPU puissant (> 8 cores) : 45s (segments plus longs = meilleure parallélisation)
        base_segment_length_ms = self.config.getint('PERFORMANCE', 'segment_length_ms', fallback=45000)
        if self.cpu_count < 4:
            self.segment_length_ms = min(base_segment_length_ms, 25000)  # 25s max pour CPU faible
            logger.info(f"⚙️ Adaptive segmentation: CPU faible ({self.cpu_count} cores) → segments de 25s")
        elif self.cpu_count <= 8:
            self.segment_length_ms = min(base_segment_length_ms, 35000)  # 35s max pour CPU moyen
            logger.info(f"⚙️ Adaptive segmentation: CPU moyen ({self.cpu_count} cores) → segments de 35s")
        else:
            self.segment_length_ms = base_segment_length_ms  # 45s pour CPU puissant
            logger.info(f"⚙️ Adaptive segmentation: CPU puissant ({self.cpu_count} cores) → segments de 45s")
        
        # Nombre optimal de workers parallèles pour transcription
        # Limiter à 1 worker par core pour éviter la surcharge mémoire
        # Whisper libère le GIL, donc ThreadPoolExecutor est optimal
        optimal_parallel_workers = min(self.cpu_count, 8)  # Max 8 workers même pour CPU très puissant
        # Permettre override via config.ini si nécessaire (désactivé pour l'instant)
        self.parallel_workers = optimal_parallel_workers
        logger.info(f"⚙️ Parallel transcription: {self.parallel_workers} worker(s) optimal")
        
        vad_enabled_str = os.environ.get(
            'VAD_ENABLED', 
            self.config.get('PERFORMANCE', 'vad_enabled')
        )
        self.vad_enabled = vad_enabled_str.lower() in ['true', '1', 't']
        
        self.beam_size = self.config.getint('PERFORMANCE', 'beam_size', fallback=1)
        self.best_of = self.config.getint('PERFORMANCE', 'best_of', fallback=1)
        self.temperature = self.config.getfloat('PERFORMANCE', 'temperature')
        
        # PATHS
        self.upload_dir = Path(os.environ.get(
            'UPLOAD_DIR',
            self.config.get('PATHS', 'upload_dir')
        ))
        
        # VAD
        self.vad_threshold = self.config.getfloat('VAD', 'threshold', fallback=0.5)
        self.vad_min_speech_duration_ms = self.config.getint('VAD', 'min_speech_duration_ms', fallback=250)
        self.vad_min_silence_duration_ms = self.config.getint('VAD', 'min_silence_duration_ms', fallback=2000)
        self.vad_speech_pad_ms = self.config.getint('VAD', 'speech_pad_ms', fallback=400)
        
        # SECURITY
        self.internal_api_key = os.environ.get(
            'INTERNAL_API_KEY', 
            self.config.get('SECURITY', 'internal_api_key')
        )
        
        if self.internal_api_key == 'CHANGE_ME_SECRET_INTERNAL_KEY_12345':
            logging.warning("⚠️ SECURITY: Internal API key is using default value. Please change it!")
        
        # LOGGING
        self.log_level = os.environ.get(
            'LOG_LEVEL', 
            self.config.get('LOGGING', 'level', fallback='INFO')
        )
        self.log_file_enabled = self.config.getboolean('LOGGING', 'file_enabled', fallback=True)
        self.log_file_path = os.environ.get(
            'LOG_FILE_PATH',
            self.config.get('LOGGING', 'file_path', fallback='logs/vocalyx-transcribe.log')
        )
        
        log_colored_str = os.environ.get(
            'LOG_COLORED', 
            self.config.get('LOGGING', 'colored', fallback='true')
        )
        self.log_colored = log_colored_str.lower() in ['true', '1', 't']
        
        # DIARIZATION
        diarization_enabled_str = os.environ.get(
            'DIARIZATION_ENABLED',
            self.config.get('DIARIZATION', 'enabled', fallback='false')
        )
        self.diarization_enabled = diarization_enabled_str.lower() in ['true', '1', 't']
        self.diarization_model = os.environ.get(
            'DIARIZATION_MODEL',
            self.config.get('DIARIZATION', 'model', fallback='pyannote/speaker-diarization-3.1')
        )
        self.diarization_model_path = os.environ.get(
            'DIARIZATION_MODEL_PATH',
            self.config.get('DIARIZATION', 'model_path', fallback='/app/models/transcribe/pyannote-speaker-diarization')
        )
        self.hf_token = os.environ.get(
            'HF_TOKEN',
            self.config.get('DIARIZATION', 'hf_token', fallback='')
        )
        
        # Paramètres d'optimisation de la diarization
        min_speakers_str = os.environ.get(
            'DIARIZATION_MIN_SPEAKERS',
            self.config.get('DIARIZATION', 'min_speakers', fallback='')
        )
        self.diarization_min_speakers = int(min_speakers_str) if min_speakers_str and min_speakers_str.isdigit() else None
        
        max_speakers_str = os.environ.get(
            'DIARIZATION_MAX_SPEAKERS',
            self.config.get('DIARIZATION', 'max_speakers', fallback='')
        )
        self.diarization_max_speakers = int(max_speakers_str) if max_speakers_str and max_speakers_str.isdigit() else None
        
        self.diarization_chunk_duration_s = self.config.getint(
            'DIARIZATION', 'chunk_duration_s', fallback=300
        )
        self.diarization_chunk_overlap_s = self.config.getint(
            'DIARIZATION', 'chunk_overlap_s', fallback=5
        )
        self.diarization_segmentation_threshold = self.config.getfloat(
            'DIARIZATION', 'segmentation_threshold', fallback=0.5
        )
        self.diarization_clustering_threshold = self.config.getfloat(
            'DIARIZATION', 'clustering_threshold', fallback=0.7
        )
        
        use_gpu_str = os.environ.get(
            'DIARIZATION_USE_GPU',
            self.config.get('DIARIZATION', 'use_gpu_if_available', fallback='true')
        )
        self.diarization_use_gpu = use_gpu_str.lower() in ['true', '1', 't']
        
        # Optimisations CPU
        num_threads_str = os.environ.get(
            'DIARIZATION_NUM_THREADS',
            self.config.get('DIARIZATION', 'num_threads', fallback='0')
        )
        self.diarization_num_threads = int(num_threads_str) if num_threads_str.isdigit() else 0
        
        reduce_resolution_str = os.environ.get(
            'DIARIZATION_REDUCE_RESOLUTION',
            self.config.get('DIARIZATION', 'reduce_resolution', fallback='false')
        )
        self.diarization_reduce_resolution = reduce_resolution_str.lower() in ['true', '1', 't']
        
        fast_mode_str = os.environ.get(
            'DIARIZATION_FAST_MODE',
            self.config.get('DIARIZATION', 'fast_mode', fallback='false')
        )
        self.diarization_fast_mode = fast_mode_str.lower() in ['true', '1', 't']
    
    def reload(self):
        """Recharge la configuration depuis le fichier"""
        self.config.read(self.config_file)
        self._load_settings()
        logging.info("🔄 Configuration reloaded")