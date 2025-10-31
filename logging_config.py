"""
logging_config.py

Configuration centralisée du logging pour Vocalyx.
Uniformise le format des logs pour tous les composants.
"""

import logging
import sys
from pathlib import Path

# Format uniforme pour tous les logs
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure le logging pour toute l'application.
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin optionnel vers un fichier de log
    """
    
    # Convertir le niveau de log
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuration de base
    handlers = []
    
    # Handler pour stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    handlers.append(console_handler)
    
    # Handler optionnel pour fichier
    if log_file:
        # Créer le répertoire si nécessaire
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    
    # Configuration globale
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True  # Override les configurations existantes
    )
    
    # Configurer les loggers spécifiques
    loggers_to_configure = [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "faster_whisper",
        "vocalyx"
    ]
    
    for logger_name in loggers_to_configure:
        log = logging.getLogger(logger_name)
        log.setLevel(numeric_level)
        # Supprimer les handlers existants pour éviter les doublons
        log.handlers.clear()
        # Ajouter nos handlers
        for handler in handlers:
            log.addHandler(handler)
        # Ne pas propager aux parents pour éviter les doublons
        log.propagate = False
    
    # Logger initial
    logger = logging.getLogger("vocalyx")
    logger.info("✅ Logging configuré avec succès")
    
    return logger


def get_uvicorn_log_config(log_level: str = "INFO"):
    """
    Retourne la configuration de logging pour Uvicorn.
    
    Args:
        log_level: Niveau de log
        
    Returns:
        dict: Configuration de logging pour Uvicorn
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": LOG_FORMAT,
                "datefmt": LOG_DATE_FORMAT,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False
            },
        },
    }


# Custom formatter avec couleurs (optionnel)
class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour le terminal"""
    
    # Codes ANSI
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Vert
        'WARNING': '\033[0;33m',  # Jaune
        'ERROR': '\033[0;31m',    # Rouge
        'CRITICAL': '\033[1;31m', # Rouge gras
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Ajouter la couleur au niveau de log
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def setup_colored_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure le logging avec couleurs pour le terminal.
    Identique à setup_logging mais avec couleurs.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = []
    
    # Handler console avec couleurs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    handlers.append(console_handler)
    
    # Handler fichier sans couleurs
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True
    )
    
    # Configurer les loggers spécifiques
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", 
                        "faster_whisper", "vocalyx"]:
        log = logging.getLogger(logger_name)
        log.setLevel(numeric_level)
        log.handlers.clear()
        for handler in handlers:
            log.addHandler(handler)
        log.propagate = False
    
    logger = logging.getLogger("vocalyx")
    logger.info("✅ Logging coloré configuré")
    
    return logger


if __name__ == "__main__":
    # Test du logging
    logger = setup_logging(log_level="INFO")
    
    logger.debug("Ceci est un message DEBUG")
    logger.info("Ceci est un message INFO")
    logger.warning("Ceci est un message WARNING")
    logger.error("Ceci est un message ERROR")
    logger.critical("Ceci est un message CRITICAL")
    
    # Test avec couleurs
    print("\n--- Test avec couleurs ---\n")
    logger = setup_colored_logging(log_level="INFO")
    
    logger.debug("Ceci est un message DEBUG")
    logger.info("Ceci est un message INFO")
    logger.warning("Ceci est un message WARNING")
    logger.error("Ceci est un message ERROR")
    logger.critical("Ceci est un message CRITICAL")