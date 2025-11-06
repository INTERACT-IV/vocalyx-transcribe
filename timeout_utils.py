"""
Utilitaires pour timeout (compatible avec multiprocessing de Celery)
"""
import threading
import functools
import logging

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Exception levée quand une opération timeout"""
    pass

def timeout(seconds):
    """
    Décorateur de timeout utilisant threading.
    Compatible avec les workers Celery en mode prefork.
    
    Args:
        seconds: Timeout en secondes
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f'Function timed out after {seconds}s')]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                logger.error(f"⏱️ Timeout: function exceeded {seconds}s")
                raise TimeoutError(f'Function timed out after {seconds}s')
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        
        return wrapper
    return decorator