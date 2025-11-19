"""
vocalyx-transcribe/config.py
Configuration du worker (adapté pour l'architecture microservices)
"""

import os
import logging
import configparser
from pathlib import Path
from typing import Set

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
        
        config['WHISPER'] = {
            'model': './models/openai-whisper-small',
            'device': 'cpu',
            'compute_type': 'int8',
            'cpu_threads': '8',
            'language': 'fr'
        }
        
        config['PERFORMANCE'] = {
            'max_workers': '2',
            'segment_length_ms': '45000',
            'vad_enabled': 'true',
            'beam_size': '5',
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
            'model_path': '/app/models/pyannote-speaker-diarization',
            'hf_token': ''
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
        self.max_workers = os.environ.get(
            'MAX_WORKERS', 
            self.config.getint('PERFORMANCE', 'max_workers')
        )
        self.segment_length_ms = self.config.getint('PERFORMANCE', 'segment_length_ms')
        
        vad_enabled_str = os.environ.get(
            'VAD_ENABLED', 
            self.config.get('PERFORMANCE', 'vad_enabled')
        )
        self.vad_enabled = vad_enabled_str.lower() in ['true', '1', 't']
        
        self.beam_size = self.config.getint('PERFORMANCE', 'beam_size')
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
            self.config.get('DIARIZATION', 'model_path', fallback='/app/models/pyannote-speaker-diarization')
        )
        self.hf_token = os.environ.get(
            'HF_TOKEN',
            self.config.get('DIARIZATION', 'hf_token', fallback='')
        )
    
    def reload(self):
        """Recharge la configuration depuis le fichier"""
        self.config.read(self.config_file)
        self._load_settings()
        logging.info("🔄 Configuration reloaded")