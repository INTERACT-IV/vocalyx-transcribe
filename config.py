"""
Gestion de la configuration de l'application
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
        
        # Créer config par défaut si n'existe pas
        if not os.path.exists(config_file):
            self._create_default_config()
        
        self.config.read(config_file)
        self._load_settings()
        
    def _create_default_config(self):
        """Crée un fichier de configuration par défaut"""
        config = configparser.ConfigParser()
        
        config['WHISPER'] = {
            'model': 'small',
            'device': 'cpu',
            'compute_type': 'int8',
            'cpu_threads': '10',
            'language': 'fr'
        }
        
        config['PERFORMANCE'] = {
            'max_workers': '2',
            'segment_length_ms': '60000',
            'vad_enabled': 'true',
            'beam_size': '5',
            'temperature': '0.0'
        }
        
        config['LIMITS'] = {
            'max_file_size_mb': '100',
            'rate_limit_per_minute': '10',
            'allowed_extensions': 'wav,mp3,m4a,flac,ogg,webm'
        }
        
        config['PATHS'] = {
            'upload_dir': './tmp_uploads',
            'database_path': 'sqlite:///./transcriptions.db',
            'templates_dir': 'templates'
        }
        
        config['VAD'] = {
            'enabled': 'true',
            'min_silence_len': '500',
            'silence_thresh': '-40',
            'vad_threshold': '0.5',
            'min_speech_duration_ms': '250',
            'min_silence_duration_ms': '500'
        }
        
        config['LOGGING'] = {
            'level': 'INFO',
            'file_enabled': 'true',
            'file_path': 'logs/vocalyx.log',
            'colored': 'false'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        logging.info(f"✅ Created default config file: {self.config_file}")
    
    def _load_settings(self):
        """Charge les paramètres dans des attributs"""
        # WHISPER
        self.model = self.config.get('WHISPER', 'model')
        self.device = self.config.get('WHISPER', 'device')
        self.compute_type = self.config.get('WHISPER', 'compute_type')
        self.cpu_threads = self.config.getint('WHISPER', 'cpu_threads')
        self.language = self.config.get('WHISPER', 'language')
        
        # PERFORMANCE
        self.max_workers = self.config.getint('PERFORMANCE', 'max_workers')
        self.segment_length_ms = self.config.getint('PERFORMANCE', 'segment_length_ms')
        self.vad_enabled = self.config.getboolean('PERFORMANCE', 'vad_enabled')
        self.beam_size = self.config.getint('PERFORMANCE', 'beam_size')
        self.temperature = self.config.getfloat('PERFORMANCE', 'temperature')
        
        # LIMITS
        self.max_file_size_mb = self.config.getint('LIMITS', 'max_file_size_mb')
        self.rate_limit = self.config.getint('LIMITS', 'rate_limit_per_minute')
        self.allowed_extensions: Set[str] = set(
            ext.strip() for ext in self.config.get('LIMITS', 'allowed_extensions').split(',')
        )
        
        # PATHS
        self.upload_dir = Path(self.config.get('PATHS', 'upload_dir'))
        self.database_path = self.config.get('PATHS', 'database_path')
        self.templates_dir = self.config.get('PATHS', 'templates_dir')
        
        # VAD
        self.vad_min_silence_len = self.config.getint('VAD', 'min_silence_len')
        self.vad_silence_thresh = self.config.getint('VAD', 'silence_thresh')
        self.vad_threshold = self.config.getfloat('VAD', 'vad_threshold')
        self.vad_min_speech_duration_ms = self.config.getint('VAD', 'min_speech_duration_ms')
        self.vad_min_silence_duration_ms = self.config.getint('VAD', 'min_silence_duration_ms')
        
        # Créer les répertoires
        self.upload_dir.mkdir(exist_ok=True)
    
    def reload(self):
        """Recharge la configuration depuis le fichier"""
        self.config.read(self.config_file)
        self._load_settings()
        logging.info("🔄 Configuration reloaded")