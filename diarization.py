"""
vocalyx-transcribe/diarization.py
Module de diarisation (identification des locuteurs) avec pyannote.audio
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("vocalyx")

# Import conditionnel de torch (nécessaire pour pyannote.audio)
try:
    import torch
    # Patcher torch.load pour PyTorch 2.6+ (weights_only=True par défaut)
    # Les modèles Pyannote nécessitent weights_only=False ou l'ajout de safe_globals
    if hasattr(torch, 'load') and not hasattr(torch, '_vocalyx_torch_load_patched'):
        original_torch_load = torch.load
        
        # Pour PyTorch 2.6+, ajouter TorchVersion aux safe_globals (méthode recommandée)
        try:
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    from torch.torch_version import TorchVersion
                    torch.serialization.add_safe_globals([TorchVersion])
                    logger.debug("✅ Added TorchVersion to torch.serialization safe_globals for PyTorch 2.6+")
                except ImportError:
                    # TorchVersion peut ne pas être importable directement
                    pass
        except (AttributeError, TypeError):
            # Si add_safe_globals n'existe pas, on utilisera weights_only=False
            pass
        
        def patched_torch_load(*args, **kwargs):
            # Si weights_only n'est pas spécifié, le définir à False pour compatibilité
            # avec les modèles Pyannote qui contiennent des objets TorchVersion
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        torch._vocalyx_torch_load_patched = True
        # Log seulement si le logger est déjà configuré (évite les logs trop tôt)
        try:
            logger.debug("✅ Patched torch.load for PyTorch 2.6+ compatibility (weights_only=False)")
        except:
            pass
except ImportError:
    torch = None
    try:
        logger.warning("⚠️ torch not installed. Diarization will not be available.")
    except:
        pass

# Mapping des modèles HuggingFace vers les chemins locaux (partagé)
_MODEL_MAPPING = {
    'pyannote/speaker-diarization-community-1': '/app/models/transcribe/pyannote-speaker-diarization-community-1',
    'pyannote/wespeaker-voxceleb-resnet34-LM': '/app/models/transcribe/pyannote-wespeaker-voxceleb-resnet34-LM',
    'pyannote/segmentation': '/app/models/transcribe/pyannote-segmentation',
    'pyannote/segmentation-3.1': '/app/models/transcribe/pyannote-segmentation',
}

def _find_local_model_file(repo_id: str, filename: str) -> Optional[Path]:
    """
    Trouve un fichier de modèle localement, en cherchant d'abord à la racine,
    puis dans le sous-dossier plda/ si c'est un fichier .npz.
    
    Returns:
        Path du fichier local s'il existe, None sinon
    """
    if repo_id not in _MODEL_MAPPING:
        return None
    
    from pathlib import Path as PathLib
    model_dir = PathLib(_MODEL_MAPPING[repo_id])
    
    # Essayer d'abord le chemin direct
    local_path = model_dir / filename
    if local_path.exists():
        return local_path
    
    # Si pas trouvé et que c'est un fichier .npz, essayer dans plda/
    if filename.endswith('.npz'):
        alt_path = model_dir / 'plda' / filename
        if alt_path.exists():
            return alt_path
    
    return None

# Patcher les bibliothèques HTTP au niveau du module (AVANT toute importation de pyannote.audio)
# Cela garantit que le patching est appliqué avant que pyannote.audio n'importe httpx
def _patch_http_for_offline_module_level():
    """Patche les bibliothèques HTTP au niveau du module pour mode offline"""
    try:
        import httpx
        from pathlib import Path as PathLib
        
        if not hasattr(httpx, '_vocalyx_module_patched'):
            # Patcher HTTPTransport.handle_request (niveau le plus bas)
            try:
                from httpx._transports.default import HTTPTransport
                original_handle_request = HTTPTransport.handle_request
                
                def patched_handle_request(self, request):
                    url = str(request.url)
                    if 'huggingface.co' in url:
                        logger.info(f"🔍 [MODULE] HTTPTransport.handle_request intercepted: {request.method} {url}")
                        import re
                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                        if match:
                            repo_id = match.group(1)
                            filename = match.group(2)
                            
                            # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                            if not isinstance(repo_id, str) or not repo_id.startswith('pyannote/'):
                                logger.debug(f"🔓 [MODULE] Non-pyannote repo {repo_id}, allowing normal request")
                                return original_handle_request(self, request)
                            
                            local_path = _find_local_model_file(repo_id, filename)
                            if local_path:
                                    logger.info(f"🔧 [MODULE] Redirecting HTTPTransport {request.method} {url} to local file: {local_path}")
                                    from httpx import Response, Headers
                                    if request.method.upper() == 'HEAD':
                                        return Response(
                                            200,
                                            headers=Headers({
                                                'content-type': 'application/octet-stream',
                                                'content-length': str(local_path.stat().st_size)
                                            })
                                        )
                                    else:
                                        with open(local_path, 'rb') as f:
                                            content = f.read()
                                        return Response(
                                            200,
                                            headers=Headers({'content-type': 'application/octet-stream'}),
                                            content=content
                                        )
                    return original_handle_request(self, request)
                
                HTTPTransport.handle_request = patched_handle_request
                httpx._vocalyx_module_patched = True
                logger.info("✅ [MODULE] Patched HTTPTransport.handle_request for offline mode")
            except (ImportError, AttributeError) as e:
                logger.warning(f"⚠️ [MODULE] Could not patch HTTPTransport: {e}")
    except ImportError:
        pass

# Appeler le patching au niveau du module
_patch_http_for_offline_module_level()

class DiarizationService:
    """
    Service de diarisation utilisant pyannote.audio pour identifier les locuteurs.
    """
    
    def __init__(self, config):
        """
        Initialise le service de diarisation.
        
        Args:
            config: Objet Config du worker
        """
        self.config = config
        self.hf_token = getattr(config, 'hf_token', None)
        self.model = None
        self.pipeline = None
        
        # Patcher les bibliothèques HTTP AVANT de charger le modèle
        self._patch_http_libraries_for_offline()
        
        self._load_model()
    
    def _fix_config_paths(self, model_path: Path):
        """
        Corrige les chemins absolus dans le config.yaml pour qu'ils pointent vers /app/models/
        au lieu de chemins de l'hôte, et remplace les références HuggingFace par des chemins locaux.
        """
        try:
            config_file = model_path / 'config.yaml'
            if not config_file.exists():
                return
            
            import re
            
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 1. Remplacer les chemins absolus de l'hôte par des chemins du container
            # Pattern pour détecter les chemins comme /home/.../shared/models/...
            pattern = r'(/home/[^/]+/[^/]+/shared/models/[^\s\'"]+)'
            
            def replace_path(match):
                old_path = match.group(1)
                # Extraire le nom du modèle (dernier segment du chemin)
                model_name = old_path.split('/')[-2] if old_path.endswith('/') else old_path.split('/')[-1]
                # Si c'est un fichier .bin, on garde le répertoire parent
                if model_name.endswith('.bin') or model_name.endswith('.pth'):
                    parent_dir = old_path.split('/')[-2]
                    new_path = f'/app/models/transcribe/{parent_dir}/{model_name}'
                else:
                    new_path = f'/app/models/transcribe/{model_name}'
                logger.info(f"🔧 Fixing path: {old_path} -> {new_path}")
                return new_path
            
            content = re.sub(pattern, replace_path, content)
            
            # 1.5. Corriger les chemins /app/models/... qui manquent /transcribe/
            # Pattern pour détecter /app/models/pyannote-... (sans /transcribe/)
            pattern_missing_transcribe = r'(/app/models/(pyannote-[^\s\'"/]+))'
            
            def fix_missing_transcribe(match):
                old_path = match.group(1)
                model_name = match.group(2)
                new_path = f'/app/models/transcribe/{model_name}'
                logger.info(f"🔧 Fixing missing /transcribe/ path: {old_path} -> {new_path}")
                return new_path
            
            content = re.sub(pattern_missing_transcribe, fix_missing_transcribe, content)
            
            # 2. Remplacer les références HuggingFace par des chemins locaux (mode offline)
            # Pattern pour détecter pyannote/... ou d'autres références HuggingFace
            hf_patterns = [
                (r'pyannote/wespeaker-voxceleb-resnet34-LM', '/app/models/transcribe/pyannote-wespeaker-voxceleb-resnet34-LM'),
                (r'pyannote/speaker-diarization-community-1', '/app/models/transcribe/pyannote-speaker-diarization-community-1'),
                (r'pyannote/segmentation', '/app/models/transcribe/pyannote-segmentation'),
                (r'pyannote/segmentation-3\.1', '/app/models/transcribe/pyannote-segmentation'),
            ]
            
            for pattern, replacement in hf_patterns:
                if re.search(pattern, content):
                    # Vérifier si le modèle local existe
                    local_path = Path(replacement)
                    if local_path.exists():
                        content = re.sub(pattern, replacement, content)
                        logger.info(f"🔧 Replaced HuggingFace reference {pattern} -> {replacement}")
                    else:
                        logger.warning(f"⚠️ HuggingFace model {pattern} referenced but local model not found at {replacement}")
            
            if content != original_content:
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info("✅ Fixed paths in config.yaml (offline mode)")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix config paths: {e}")
    
    def _patch_http_libraries_for_offline(self):
        """
        Patche les bibliothèques HTTP (httpx, requests) et huggingface_hub pour intercepter 
        les requêtes HuggingFace et les rediriger vers les modèles locaux. 
        Doit être appelé AVANT l'import de pyannote.audio.
        """
        from pathlib import Path as PathLib
        
        # Patcher huggingface_hub directement (plusieurs fonctions possibles)
        try:
            import huggingface_hub
            if not hasattr(huggingface_hub, '_vocalyx_patched'):
                # Patcher hf_hub_download
                if hasattr(huggingface_hub, 'hf_hub_download'):
                    original_hf_hub_download = huggingface_hub.hf_hub_download
                    
                    def patched_hf_hub_download(repo_id, filename, **kwargs):
                        logger.info(f"🔍 huggingface_hub.hf_hub_download called: repo_id={repo_id}, filename={filename}")
                        
                        # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                        # (pour permettre Whisper et autres modèles de fonctionner normalement)
                        if isinstance(repo_id, str) and not repo_id.startswith('pyannote/'):
                            # Ce n'est pas un modèle pyannote, laisser passer sans modification
                            logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal download")
                            return original_hf_hub_download(repo_id, filename, **kwargs)
                        
                        # Détecter si repo_id est en fait un chemin local (commence par / ou ./)
                        if isinstance(repo_id, str) and (repo_id.startswith('/') or repo_id.startswith('./')):
                            # C'est un chemin local, pas un repo_id HuggingFace
                            local_file_path = Path(repo_id)
                            if local_file_path.exists():
                                logger.info(f"🔧 Detected local file path, returning directly: {local_file_path}")
                                return str(local_file_path)
                            else:
                                # Si le chemin n'existe pas, essayer avec filename comme suffixe
                                local_file_path = Path(repo_id) / filename if filename else Path(repo_id)
                                if local_file_path.exists():
                                    logger.info(f"🔧 Detected local file path with filename, returning: {local_file_path}")
                                    return str(local_file_path)
                                
                                # Si toujours pas trouvé, essayer de corriger le chemin en ajoutant /transcribe/
                                # Ex: /app/models/pyannote-segmentation -> /app/models/transcribe/pyannote-segmentation
                                if '/app/models/' in repo_id and '/transcribe/' not in repo_id:
                                    # Extraire le nom du modèle du chemin
                                    parts = repo_id.replace('/app/models/', '').split('/')
                                    if parts:
                                        model_name = parts[0]
                                        corrected_path = Path(f'/app/models/transcribe/{model_name}')
                                        if filename:
                                            corrected_file = corrected_path / filename
                                        else:
                                            corrected_file = corrected_path
                                        
                                        if corrected_file.exists():
                                            logger.info(f"🔧 Corrected path from {repo_id} to {corrected_file}")
                                            return str(corrected_file)
                                
                                logger.warning(f"⚠️ Local path not found: {repo_id}")
                                raise FileNotFoundError(f"Local model file not found: {repo_id}")
                        
                        # Traiter comme un repo_id pyannote HuggingFace
                        local_path = _find_local_model_file(repo_id, filename)
                        if local_path:
                            logger.info(f"🔧 Redirecting hf_hub_download {repo_id}/{filename} to local file: {local_path}")
                            return str(local_path)
                        else:
                            logger.warning(f"⚠️ File {filename} not found for {repo_id}")
                        # Sinon, forcer le mode offline (seulement pour pyannote)
                        kwargs['local_files_only'] = True
                        try:
                            return original_hf_hub_download(repo_id, filename, **kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to download {repo_id}/{filename} in offline mode: {e}")
                            raise
                    
                    huggingface_hub.hf_hub_download = patched_hf_hub_download
                
                # Patcher aussi try_to_load_from_cache qui est utilisé en interne
                try:
                    from huggingface_hub import file_download
                    original_file_download = file_download
                    
                    def patched_file_download(repo_id, filename, **kwargs):
                        logger.info(f"🔍 huggingface_hub.file_download called: repo_id={repo_id}, filename={filename}")
                        
                        # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                        if isinstance(repo_id, str) and not repo_id.startswith('pyannote/'):
                            logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal download")
                            return original_file_download(repo_id, filename, **kwargs)
                        
                        local_path = _find_local_model_file(repo_id, filename)
                        if local_path:
                            logger.info(f"🔧 Redirecting file_download {repo_id}/{filename} to local file: {local_path}")
                            return str(local_path)
                        # Sinon, forcer le mode offline (seulement pour pyannote)
                        kwargs['local_files_only'] = True
                        try:
                            return original_file_download(repo_id, filename, **kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed file_download {repo_id}/{filename} in offline mode: {e}")
                            raise
                    
                    huggingface_hub.file_download = patched_file_download
                    logger.info("✅ Patched huggingface_hub.file_download")
                except (ImportError, AttributeError):
                    pass
                
                huggingface_hub._vocalyx_patched = True
                logger.info("✅ Patched huggingface_hub functions for offline mode (early)")
        except (ImportError, AttributeError) as e:
            logger.warning(f"⚠️ Could not patch huggingface_hub: {e}")
        
        # Patcher httpx au niveau du transport pour intercepter toutes les requêtes
        try:
            import httpx
            
            if not hasattr(httpx, '_vocalyx_patched'):
                # Patcher httpx.Client.request directement
                original_client_request = httpx.Client.request
                
                def patched_client_request(self, method, url, **kwargs):
                    if isinstance(url, str) and 'huggingface.co' in url:
                        logger.info(f"🔍 httpx.Client.request intercepted: {method} {url}")
                        # Extraire repo_id et filename de l'URL
                        import re
                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                        if match:
                            repo_id = match.group(1)
                            filename = match.group(2)
                            
                            # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                            if not isinstance(repo_id, str) or not repo_id.startswith('pyannote/'):
                                logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal request")
                                return original_client_request(self, method, url, **kwargs)
                            
                            local_path = _find_local_model_file(repo_id, filename)
                            if local_path:
                                    logger.info(f"🔧 Redirecting httpx.Client.request {method} {url} to local file: {local_path}")
                                    # Retourner un objet mock
                                    if method.upper() == 'HEAD':
                                        class MockHttpxHeadResponse:
                                            def __init__(self, file_path):
                                                self.status_code = 200
                                                self.headers = httpx.Headers({
                                                    'content-type': 'application/octet-stream',
                                                    'content-length': str(local_path.stat().st_size)
                                                })
                                            
                                            def raise_for_status(self):
                                                pass
                                            
                                            def close(self):
                                                pass
                                            
                                            def __enter__(self):
                                                return self
                                            
                                            def __exit__(self, *args):
                                                pass
                                            
                                            async def __aenter__(self):
                                                return self
                                            
                                            async def __aexit__(self, *args):
                                                pass
                                        
                                        return MockHttpxHeadResponse(local_path)
                                    else:
                                        class MockHttpxResponse:
                                            def __init__(self, file_path):
                                                self.file_path = file_path
                                                self.status_code = 200
                                                self.headers = httpx.Headers({'content-type': 'application/octet-stream'})
                                                self._content = None
                                            
                                            def read(self):
                                                if self._content is None:
                                                    with open(self.file_path, 'rb') as f:
                                                        self._content = f.read()
                                                return self._content
                                            
                                            @property
                                            def content(self):
                                                return self.read()
                                            
                                            def raise_for_status(self):
                                                pass
                                            
                                            def json(self):
                                                import json
                                                with open(self.file_path, 'r') as f:
                                                    return json.load(f)
                                            
                                            def text(self):
                                                with open(self.file_path, 'r', encoding='utf-8') as f:
                                                    return f.read()
                                            
                                            def iter_bytes(self, chunk_size=None):
                                                with open(self.file_path, 'rb') as f:
                                                    while True:
                                                        chunk = f.read(chunk_size or 8192)
                                                        if not chunk:
                                                            break
                                                        yield chunk
                                            
                                            def close(self):
                                                pass
                                            
                                            def __enter__(self):
                                                return self
                                            
                                            def __exit__(self, *args):
                                                pass
                                            
                                            async def __aenter__(self):
                                                return self
                                            
                                            async def __aexit__(self, *args):
                                                pass
                                        
                                        return MockHttpxResponse(local_path)
                    # Sinon, utiliser la méthode originale
                    return original_client_request(self, method, url, **kwargs)
                
                httpx.Client.request = patched_client_request
                
                # Patcher aussi httpx.AsyncClient si disponible
                try:
                    original_async_client_request = httpx.AsyncClient.request
                    
                    async def patched_async_client_request(self, method, url, **kwargs):
                        if isinstance(url, str) and 'huggingface.co' in url:
                            logger.info(f"🔍 httpx.AsyncClient.request intercepted: {method} {url}")
                            # Même logique que pour Client
                            import re
                            match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                            if match:
                                repo_id = match.group(1)
                                filename = match.group(2)
                                
                                # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                                if not isinstance(repo_id, str) or not repo_id.startswith('pyannote/'):
                                    logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal request")
                                    return await original_async_client_request(self, method, url, **kwargs)
                                
                                local_path = _find_local_model_file(repo_id, filename)
                                if local_path:
                                        logger.info(f"🔧 Redirecting AsyncClient.request {method} {url} to local file: {local_path}")
                                        # Retourner un objet mock async
                                        if method.upper() == 'HEAD':
                                            class MockAsyncHttpxHeadResponse:
                                                def __init__(self, file_path):
                                                    self.status_code = 200
                                                    self.headers = httpx.Headers({
                                                        'content-type': 'application/octet-stream',
                                                        'content-length': str(local_path.stat().st_size)
                                                    })
                                                
                                                def raise_for_status(self):
                                                    pass
                                                
                                                async def __aenter__(self):
                                                    return self
                                                
                                                async def __aexit__(self, *args):
                                                    pass
                                            
                                            return MockAsyncHttpxHeadResponse(local_path)
                                        else:
                                            class MockAsyncHttpxResponse:
                                                def __init__(self, file_path):
                                                    self.file_path = file_path
                                                    self.status_code = 200
                                                    self.headers = httpx.Headers({'content-type': 'application/octet-stream'})
                                                
                                                async def aread(self):
                                                    with open(self.file_path, 'rb') as f:
                                                        return f.read()
                                                
                                                @property
                                                async def content(self):
                                                    return await self.aread()
                                                
                                                def raise_for_status(self):
                                                    pass
                                                
                                                async def __aenter__(self):
                                                    return self
                                                
                                                async def __aexit__(self, *args):
                                                    pass
                                            
                                            return MockAsyncHttpxResponse(local_path)
                        return await original_async_client_request(self, method, url, **kwargs)
                    
                    httpx.AsyncClient.request = patched_async_client_request
                    logger.info("✅ Patched httpx.AsyncClient.request")
                except (AttributeError, TypeError):
                    pass
                
                # Patcher aussi le transport HTTP pour intercepter au niveau le plus bas
                try:
                    from httpx._transports.default import HTTPTransport
                    original_handle_request = HTTPTransport.handle_request
                    
                    def patched_handle_request(self, request):
                        url = str(request.url)
                        if 'huggingface.co' in url:
                            logger.info(f"🔍 HTTPTransport.handle_request intercepted: {request.method} {url}")
                            # Extraire repo_id et filename
                            import re
                            match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                            if match:
                                repo_id = match.group(1)
                                filename = match.group(2)
                                
                                # Si ce n'est PAS un modèle pyannote, laisser passer l'appel original
                                if not isinstance(repo_id, str) or not repo_id.startswith('pyannote/'):
                                    logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal request")
                                    return original_handle_request(self, request)
                                
                                # Chercher le fichier local pour les modèles pyannote
                                local_path = _find_local_model_file(repo_id, filename)
                                if local_path:
                                    logger.info(f"🔧 Redirecting HTTPTransport {request.method} {url} to local file: {local_path}")
                                    # Créer une réponse mock
                                    from httpx import Response, Headers
                                    if request.method.upper() == 'HEAD':
                                        return Response(
                                            200,
                                            headers=Headers({
                                                'content-type': 'application/octet-stream',
                                                'content-length': str(local_path.stat().st_size)
                                            })
                                        )
                                    else:
                                        with open(local_path, 'rb') as f:
                                            content = f.read()
                                        return Response(
                                            200,
                                            headers=Headers({'content-type': 'application/octet-stream'}),
                                            content=content
                                        )
                        # Sinon, utiliser la méthode originale
                        return original_handle_request(self, request)
                    
                    HTTPTransport.handle_request = patched_handle_request
                    logger.info("✅ Patched HTTPTransport.handle_request for offline mode")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"⚠️ Could not patch HTTPTransport: {e}")
                
                httpx._vocalyx_patched = True
                logger.info("✅ Patched httpx.Client.request for offline mode (early)")
        except ImportError:
            logger.warning("⚠️ httpx not available for patching")
        except Exception as e:
            logger.warning(f"⚠️ Failed to patch httpx.Client.request: {e}")
    
    def _load_model(self):
        """Charge le modèle de diarisation pyannote.audio"""
        try:
            from pyannote.audio import Pipeline
            from pathlib import Path
            
            logger.info("🚀 Loading pyannote.audio diarization model...")
            
            # Vérifier d'abord si un modèle local est disponible dans /app/models/
            model_path = getattr(self.config, 'diarization_model_path', None)
            logger.info(f"🔍 Checking diarization model path: {model_path}")
            
            if not model_path:
                # Par défaut, chercher dans /app/models/transcribe/pyannote-speaker-diarization (chemin dans le container Docker)
                default_model_path = Path('/app/models/transcribe/pyannote-speaker-diarization')
                logger.info(f"🔍 Checking default path: {default_model_path}")
                logger.info(f"🔍 Path exists: {default_model_path.exists()}")
                if default_model_path.exists():
                    config_file = default_model_path / 'config.yaml'
                    logger.info(f"🔍 Config file exists: {config_file.exists()}")
                    if config_file.exists():
                        model_path = str(default_model_path)
                        logger.info(f"📁 Found local model at: {model_path}")
            
            if model_path:
                model_path_obj = Path(model_path)
                logger.info(f"🔍 Final model path: {model_path}")
                logger.info(f"🔍 Path exists: {model_path_obj.exists()}")
                if model_path_obj.exists():
                    # Charger depuis un chemin local
                    logger.info(f"📥 Loading model from local path: {model_path}")
                    try:
                        # Corriger les chemins absolus dans le config.yaml si nécessaire
                        self._fix_config_paths(model_path_obj)
                        
                        # Désactiver les téléchargements HuggingFace pour mode offline
                        import os
                        original_hf_home = os.environ.get('HF_HOME')
                        original_huggingface_hub_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
                        
                        # Forcer l'utilisation locale uniquement
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        
                        # Forcer PyTorch à utiliser weights_only=False pour compatibilité avec Pyannote
                        # (PyTorch 2.6+ utilise weights_only=True par défaut)
                        os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
                        
                        # Mapper les modèles HuggingFace vers les chemins locaux
                        # Cela permet d'intercepter les appels aux modèles gated
                        model_mapping = {
                            'pyannote/speaker-diarization-community-1': '/app/models/transcribe/pyannote-speaker-diarization-community-1',
                            'pyannote/wespeaker-voxceleb-resnet34-LM': '/app/models/transcribe/pyannote-wespeaker-voxceleb-resnet34-LM',
                            'pyannote/segmentation': '/app/models/transcribe/pyannote-segmentation',
                            'pyannote/segmentation-3.1': '/app/models/transcribe/pyannote-segmentation',
                        }
                        
                        # Patcher temporairement le système de chargement HuggingFace
                        try:
                            from huggingface_hub import snapshot_download
                            from pathlib import Path as PathLib
                            
                            original_snapshot_download = snapshot_download
                            
                            def patched_snapshot_download(repo_id, **kwargs):
                                # Si le modèle est dans notre mapping, utiliser le chemin local
                                if repo_id in model_mapping:
                                    local_path = PathLib(model_mapping[repo_id])
                                    if local_path.exists():
                                        logger.info(f"🔧 Redirecting HuggingFace model {repo_id} to local path: {local_path}")
                                        return str(local_path)
                                # Sinon, utiliser la fonction originale mais en mode offline
                                kwargs['local_files_only'] = True
                                return original_snapshot_download(repo_id, **kwargs)
                            
                            # Patcher le module huggingface_hub
                            import huggingface_hub
                            huggingface_hub.snapshot_download = patched_snapshot_download
                            
                            # Patcher aussi hf_hub_download si disponible (utilisé pour télécharger des fichiers individuels)
                            try:
                                from huggingface_hub import hf_hub_download
                                original_hf_hub_download = hf_hub_download
                                
                                def patched_hf_hub_download(repo_id, filename, **kwargs):
                                    logger.info(f"🔍 hf_hub_download called: repo_id={repo_id}, filename={filename}")
                                    
                                    # Détecter si repo_id est en fait un chemin local (commence par / ou ./)
                                    if isinstance(repo_id, str) and (repo_id.startswith('/') or repo_id.startswith('./')):
                                        # C'est un chemin local, pas un repo_id HuggingFace
                                        local_file_path = PathLib(repo_id)
                                        if local_file_path.exists():
                                            logger.info(f"🔧 Detected local file path, returning directly: {local_file_path}")
                                            return str(local_file_path)
                                        else:
                                            # Si le chemin n'existe pas, essayer avec filename comme suffixe
                                            if filename:
                                                local_file_path = PathLib(repo_id) / filename
                                                if local_file_path.exists():
                                                    logger.info(f"🔧 Detected local file path with filename, returning: {local_file_path}")
                                                    return str(local_file_path)
                                            
                                            # Si toujours pas trouvé, essayer de corriger le chemin en ajoutant /transcribe/
                                            # Ex: /app/models/pyannote-segmentation -> /app/models/transcribe/pyannote-segmentation
                                            if '/app/models/' in repo_id and '/transcribe/' not in repo_id:
                                                # Extraire le nom du modèle du chemin
                                                parts = repo_id.replace('/app/models/', '').split('/')
                                                if parts:
                                                    model_name = parts[0]
                                                    corrected_path = PathLib(f'/app/models/transcribe/{model_name}')
                                                    if filename:
                                                        corrected_file = corrected_path / filename
                                                    else:
                                                        corrected_file = corrected_path
                                                    
                                                    if corrected_file.exists():
                                                        logger.info(f"🔧 Corrected path from {repo_id} to {corrected_file}")
                                                        return str(corrected_file)
                                            
                                            logger.warning(f"⚠️ Local path not found: {repo_id}")
                                            raise FileNotFoundError(f"Local model file not found: {repo_id}")
                                    
                                    # Sinon, traiter comme un repo_id HuggingFace normal
                                    # Si le modèle est dans notre mapping, construire le chemin local
                                    if repo_id in model_mapping:
                                        local_path = PathLib(model_mapping[repo_id]) / filename
                                        logger.info(f"🔍 Checking local path: {local_path} (exists: {local_path.exists()})")
                                        if local_path.exists():
                                            logger.info(f"🔧 Redirecting hf_hub_download {repo_id}/{filename} to local path: {local_path}")
                                            return str(local_path)
                                        else:
                                            logger.warning(f"⚠️ File {filename} not found at {local_path}")
                                            # Lister les fichiers disponibles pour debug
                                            model_dir = PathLib(model_mapping[repo_id])
                                            if model_dir.exists():
                                                logger.info(f"🔍 Available files in {model_dir}: {list(model_dir.rglob('*'))}")
                                    else:
                                        logger.warning(f"⚠️ repo_id {repo_id} not in model_mapping: {list(model_mapping.keys())}")
                                        # Si ce n'est pas un modèle pyannote, laisser passer normalement
                                        if not isinstance(repo_id, str) or not repo_id.startswith('pyannote/'):
                                            logger.debug(f"🔓 Non-pyannote repo {repo_id}, allowing normal download")
                                            return original_hf_hub_download(repo_id, filename, **kwargs)
                                    # Sinon, utiliser la fonction originale mais en mode offline (seulement pour pyannote)
                                    kwargs['local_files_only'] = True
                                    try:
                                        return original_hf_hub_download(repo_id, filename, **kwargs)
                                    except Exception as e:
                                        logger.error(f"❌ Failed to download {repo_id}/{filename} in offline mode: {e}")
                                        raise
                                
                                huggingface_hub.hf_hub_download = patched_hf_hub_download
                                logger.info("✅ Patched hf_hub_download for offline mode")
                                
                                # Patcher aussi cached_download si disponible (ancienne API)
                                try:
                                    from huggingface_hub import cached_download
                                    original_cached_download = cached_download
                                    
                                    def patched_cached_download(url_or_filename, **kwargs):
                                        logger.info(f"🔍 cached_download called: {url_or_filename}")
                                        # Si c'est une URL HuggingFace, essayer de la mapper
                                        if isinstance(url_or_filename, str) and 'huggingface.co' in url_or_filename:
                                            # Extraire repo_id et filename de l'URL
                                            # Format: https://huggingface.co/repo_id/resolve/main/filename
                                            import re
                                            match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url_or_filename)
                                            if match:
                                                repo_id = match.group(1)
                                                filename = match.group(2)
                                                if repo_id in model_mapping:
                                                    local_path = PathLib(model_mapping[repo_id]) / filename
                                                    if local_path.exists():
                                                        logger.info(f"🔧 Redirecting cached_download {url_or_filename} to local path: {local_path}")
                                                        return str(local_path)
                                        # Sinon, utiliser la fonction originale
                                        return original_cached_download(url_or_filename, **kwargs)
                                    
                                    huggingface_hub.cached_download = patched_cached_download
                                    logger.info("✅ Patched cached_download for offline mode")
                                except (ImportError, AttributeError):
                                    pass
                                
                            except ImportError:
                                logger.warning("⚠️ hf_hub_download not available for patching")
                            
                            # Patcher requests pour intercepter les requêtes HTTP directes vers HuggingFace
                            try:
                                import requests
                                original_requests_get = requests.get
                                original_requests_head = requests.head
                                
                                def patched_requests_get(url, **kwargs):
                                    if isinstance(url, str) and 'huggingface.co' in url:
                                        logger.info(f"🔍 requests.get intercepted: {url}")
                                        # Extraire repo_id et filename de l'URL
                                        import re
                                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                                        if match:
                                            repo_id = match.group(1)
                                            filename = match.group(2)
                                            if repo_id in model_mapping:
                                                local_path = PathLib(model_mapping[repo_id]) / filename
                                                if local_path.exists():
                                                    logger.info(f"🔧 Redirecting requests.get {url} to local file: {local_path}")
                                                    # Retourner un objet mock qui simule une réponse HTTP
                                                    class MockResponse:
                                                        def __init__(self, file_path):
                                                            self.file_path = file_path
                                                            self.status_code = 200
                                                            self.content = open(file_path, 'rb').read()
                                                            self.headers = {'Content-Type': 'application/octet-stream'}
                                                        
                                                        def iter_content(self, chunk_size=None):
                                                            with open(self.file_path, 'rb') as f:
                                                                while True:
                                                                    chunk = f.read(chunk_size or 8192)
                                                                    if not chunk:
                                                                        break
                                                                    yield chunk
                                                        
                                                        def raise_for_status(self):
                                                            pass
                                                        
                                                        def json(self):
                                                            import json
                                                            with open(self.file_path, 'r') as f:
                                                                return json.load(f)
                                                        
                                                        def text(self):
                                                            with open(self.file_path, 'r', encoding='utf-8') as f:
                                                                return f.read()
                                                        
                                                        def close(self):
                                                            pass
                                                        
                                                        def __enter__(self):
                                                            return self
                                                        
                                                        def __exit__(self, *args):
                                                            pass
                                                    
                                                    return MockResponse(local_path)
                                    # Sinon, utiliser la fonction originale
                                    return original_requests_get(url, **kwargs)
                                
                                def patched_requests_head(url, **kwargs):
                                    if isinstance(url, str) and 'huggingface.co' in url:
                                        logger.info(f"🔍 requests.head intercepted: {url}")
                                        # Extraire repo_id et filename de l'URL
                                        import re
                                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                                        if match:
                                            repo_id = match.group(1)
                                            filename = match.group(2)
                                            if repo_id in model_mapping:
                                                local_path = PathLib(model_mapping[repo_id]) / filename
                                                if local_path.exists():
                                                    logger.info(f"🔧 Redirecting requests.head {url} to local file: {local_path}")
                                                    # Retourner un objet mock pour HEAD
                                                    class MockHeadResponse:
                                                        def __init__(self, file_path):
                                                            self.status_code = 200
                                                            self.headers = {
                                                                'Content-Type': 'application/octet-stream',
                                                                'Content-Length': str(local_path.stat().st_size)
                                                            }
                                                        
                                                        def raise_for_status(self):
                                                            pass
                                                        
                                                        def close(self):
                                                            pass
                                                        
                                                        def __enter__(self):
                                                            return self
                                                        
                                                        def __exit__(self, *args):
                                                            pass
                                                    
                                                    return MockHeadResponse(local_path)
                                    # Sinon, utiliser la fonction originale
                                    return original_requests_head(url, **kwargs)
                                
                                requests.get = patched_requests_get
                                requests.head = patched_requests_head
                                logger.info("✅ Patched requests.get and requests.head for offline mode")
                                
                            except ImportError:
                                logger.warning("⚠️ requests not available for patching")
                            
                            # Patcher aussi httpx (utilisé par huggingface_hub moderne)
                            try:
                                import httpx
                                original_httpx_get = httpx.get
                                original_httpx_head = httpx.head
                                
                                def patched_httpx_get(url, **kwargs):
                                    if isinstance(url, str) and 'huggingface.co' in url:
                                        logger.info(f"🔍 httpx.get intercepted: {url}")
                                        # Extraire repo_id et filename de l'URL
                                        import re
                                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                                        if match:
                                            repo_id = match.group(1)
                                            filename = match.group(2)
                                            if repo_id in model_mapping:
                                                local_path = PathLib(model_mapping[repo_id]) / filename
                                                if local_path.exists():
                                                    logger.info(f"🔧 Redirecting httpx.get {url} to local file: {local_path}")
                                                    # Retourner un objet mock similaire à httpx.Response
                                                    class MockHttpxResponse:
                                                        def __init__(self, file_path):
                                                            self.file_path = file_path
                                                            self.status_code = 200
                                                            self.headers = httpx.Headers({'content-type': 'application/octet-stream'})
                                                            self._content = None
                                                        
                                                        def read(self):
                                                            if self._content is None:
                                                                with open(self.file_path, 'rb') as f:
                                                                    self._content = f.read()
                                                            return self._content
                                                        
                                                        @property
                                                        def content(self):
                                                            return self.read()
                                                        
                                                        def raise_for_status(self):
                                                            pass
                                                        
                                                        def json(self):
                                                            import json
                                                            with open(self.file_path, 'r') as f:
                                                                return json.load(f)
                                                        
                                                        def text(self):
                                                            with open(self.file_path, 'r', encoding='utf-8') as f:
                                                                return f.read()
                                                        
                                                        def iter_bytes(self, chunk_size=None):
                                                            with open(self.file_path, 'rb') as f:
                                                                while True:
                                                                    chunk = f.read(chunk_size or 8192)
                                                                    if not chunk:
                                                                        break
                                                                    yield chunk
                                                        
                                                        def close(self):
                                                            pass
                                                        
                                                        def __enter__(self):
                                                            return self
                                                        
                                                        def __exit__(self, *args):
                                                            pass
                                                        
                                                        async def __aenter__(self):
                                                            return self
                                                        
                                                        async def __aexit__(self, *args):
                                                            pass
                                                    
                                                    return MockHttpxResponse(local_path)
                                    # Sinon, utiliser la fonction originale
                                    return original_httpx_get(url, **kwargs)
                                
                                def patched_httpx_head(url, **kwargs):
                                    if isinstance(url, str) and 'huggingface.co' in url:
                                        logger.info(f"🔍 httpx.head intercepted: {url}")
                                        # Extraire repo_id et filename de l'URL
                                        import re
                                        match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                                        if match:
                                            repo_id = match.group(1)
                                            filename = match.group(2)
                                            if repo_id in model_mapping:
                                                local_path = PathLib(model_mapping[repo_id]) / filename
                                                if local_path.exists():
                                                    logger.info(f"🔧 Redirecting httpx.head {url} to local file: {local_path}")
                                                    # Retourner un objet mock pour HEAD
                                                    class MockHttpxHeadResponse:
                                                        def __init__(self, file_path):
                                                            self.status_code = 200
                                                            self.headers = httpx.Headers({
                                                                'content-type': 'application/octet-stream',
                                                                'content-length': str(local_path.stat().st_size)
                                                            })
                                                        
                                                        def raise_for_status(self):
                                                            pass
                                                        
                                                        def close(self):
                                                            pass
                                                        
                                                        def __enter__(self):
                                                            return self
                                                        
                                                        def __exit__(self, *args):
                                                            pass
                                                        
                                                        async def __aenter__(self):
                                                            return self
                                                        
                                                        async def __aexit__(self, *args):
                                                            pass
                                                    
                                                    return MockHttpxHeadResponse(local_path)
                                    # Sinon, utiliser la fonction originale
                                    return original_httpx_head(url, **kwargs)
                                
                                httpx.get = patched_httpx_get
                                httpx.head = patched_httpx_head
                                
                                # Patcher aussi httpx.Client qui est utilisé par huggingface_hub
                                original_httpx_client_init = httpx.Client.__init__
                                original_httpx_client_request = httpx.Client.request
                                
                                def patched_httpx_client_init(self, *args, **kwargs):
                                    """
                                    Initialise httpx.Client en patchant la méthode request
                                    pour rediriger les appels vers HuggingFace en local.
                                    
                                    ⚠️ Important : on utilise toujours la méthode
                                    originale stockée dans original_httpx_client_request
                                    (capturée dans la closure), pour éviter toute
                                    récursion si __init__ est appelé plusieurs fois
                                    sur la même instance.
                                    """
                                    # Appeler l'__init__ original
                                    original_httpx_client_init(self, *args, **kwargs)
                                    
                                    def patched_request(method, url, **req_kwargs):
                                        if isinstance(url, str) and 'huggingface.co' in url:
                                            logger.info(f"🔍 httpx.Client.request intercepted: {method} {url}")
                                            # Extraire repo_id et filename de l'URL
                                            import re
                                            match = re.search(r'huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)', url)
                                            if match:
                                                repo_id = match.group(1)
                                                filename = match.group(2)
                                                if repo_id in model_mapping:
                                                    local_path = PathLib(model_mapping[repo_id]) / filename
                                                    if local_path.exists():
                                                        logger.info(f"🔧 Redirecting httpx.Client.request {method} {url} to local file: {local_path}")
                                                        # Retourner un objet mock
                                                        if method.upper() == 'HEAD':
                                                            class MockHttpxHeadResponse:
                                                                def __init__(self, file_path):
                                                                    self.status_code = 200
                                                                    self.headers = httpx.Headers({
                                                                        'content-type': 'application/octet-stream',
                                                                        'content-length': str(local_path.stat().st_size)
                                                                    })
                                                                
                                                                def raise_for_status(self):
                                                                    pass
                                                                
                                                                def close(self):
                                                                    pass
                                                                
                                                                def __enter__(self):
                                                                    return self
                                                                
                                                                def __exit__(self, *args):
                                                                    pass
                                                                
                                                                async def __aenter__(self):
                                                                    return self
                                                                
                                                                async def __aexit__(self, *args):
                                                                    pass
                                                            
                                                            return MockHttpxHeadResponse(local_path)
                                                        else:
                                                            class MockHttpxResponse:
                                                                def __init__(self, file_path):
                                                                    self.file_path = file_path
                                                                    self.status_code = 200
                                                                    self.headers = httpx.Headers({'content-type': 'application/octet-stream'})
                                                                    self._content = None
                                                                
                                                                def read(self):
                                                                    if self._content is None:
                                                                        with open(self.file_path, 'rb') as f:
                                                                            self._content = f.read()
                                                                    return self._content
                                                                
                                                                @property
                                                                def content(self):
                                                                    return self.read()
                                                                
                                                                def raise_for_status(self):
                                                                    pass
                                                                
                                                                def json(self):
                                                                    import json
                                                                    with open(self.file_path, 'r') as f:
                                                                        return json.load(f)
                                                                
                                                                def text(self):
                                                                    with open(self.file_path, 'r', encoding='utf-8') as f:
                                                                        return f.read()
                                                                
                                                                def iter_bytes(self, chunk_size=None):
                                                                    with open(self.file_path, 'rb') as f:
                                                                        while True:
                                                                            chunk = f.read(chunk_size or 8192)
                                                                            if not chunk:
                                                                                break
                                                                            yield chunk
                                                                
                                                                def close(self):
                                                                    pass
                                                                
                                                                def __enter__(self):
                                                                    return self
                                                                
                                                                def __exit__(self, *args):
                                                                    pass
                                                                
                                                                async def __aenter__(self):
                                                                    return self
                                                                
                                                                async def __aexit__(self, *args):
                                                                    pass
                                                            
                                                            return MockHttpxResponse(local_path)
                                        # Sinon, utiliser la méthode originale (non patchée)
                                        # en appelant directement original_httpx_client_request
                                        return original_httpx_client_request(self, method, url, **req_kwargs)
                                    
                                    self.request = patched_request
                                
                                httpx.Client.__init__ = patched_httpx_client_init
                                logger.info("✅ Patched httpx.get, httpx.head, and httpx.Client for offline mode")
                                
                            except ImportError:
                                logger.warning("⚠️ httpx not available for patching")
                            
                        except ImportError as e:
                            logger.warning(f"⚠️ Could not patch HuggingFace loaders: {e}, will rely on environment variables")
                        
                        try:
                            # S'assurer que torch.load est bien patché avant de charger le modèle
                            if torch is not None:
                                # Vérifier que le patch est appliqué
                                if not hasattr(torch, '_vocalyx_torch_load_patched'):
                                    # Re-appliquer le patch si nécessaire
                                    original_torch_load = torch.load
                                    def patched_torch_load(*args, **kwargs):
                                        if 'weights_only' not in kwargs:
                                            kwargs['weights_only'] = False
                                        return original_torch_load(*args, **kwargs)
                                    torch.load = patched_torch_load
                                    torch._vocalyx_torch_load_patched = True
                                    logger.info("✅ Re-applied torch.load patch before model loading")
                                
                                # Ajouter TorchVersion aux safe_globals si disponible (PyTorch 2.6+)
                                try:
                                    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                                        try:
                                            from torch.torch_version import TorchVersion
                                            # Vérifier si déjà ajouté pour éviter les doublons
                                            if not hasattr(torch.serialization, '_vocalyx_safe_globals_added'):
                                                torch.serialization.add_safe_globals([TorchVersion])
                                                torch.serialization._vocalyx_safe_globals_added = True
                                                logger.info("✅ Added TorchVersion to safe_globals for PyTorch 2.6+")
                                        except ImportError:
                                            logger.warning("⚠️ Could not import TorchVersion, will use weights_only=False")
                                except (AttributeError, TypeError) as e:
                                    logger.debug(f"⚠️ add_safe_globals not available: {e}")
                            
                            # Essayer avec local_files_only (nouvelle API)
                            # Utiliser un context manager pour s'assurer que torch.load utilise weights_only=False
                            try:
                                # Créer un context manager pour patcher torch.load temporairement
                                import contextlib
                                
                                @contextlib.contextmanager
                                def torch_load_context():
                                    """Context manager pour forcer weights_only=False"""
                                    if torch is not None and hasattr(torch, 'load'):
                                        original_load = torch.load
                                        def safe_load(*args, **kwargs):
                                            kwargs['weights_only'] = False
                                            return original_load(*args, **kwargs)
                                        torch.load = safe_load
                                        try:
                                            yield
                                        finally:
                                            torch.load = original_load
                                    else:
                                        yield
                                
                                with torch_load_context():
                                    try:
                                        self.pipeline = Pipeline.from_pretrained(
                                            model_path,
                                            local_files_only=True
                                        )
                                    except TypeError:
                                        # Fallback pour les anciennes versions qui n'ont pas local_files_only
                                        self.pipeline = Pipeline.from_pretrained(model_path)
                            except Exception as load_error:
                                # Si le context manager échoue, essayer sans
                                logger.warning(f"⚠️ Context manager failed, trying direct load: {load_error}")
                                try:
                                    self.pipeline = Pipeline.from_pretrained(
                                        model_path,
                                        local_files_only=True
                                    )
                                except TypeError:
                                    self.pipeline = Pipeline.from_pretrained(model_path)
                            logger.info("✅ Model loaded from local path (offline mode)")
                        finally:
                            # Restaurer les fonctions originales
                            try:
                                import huggingface_hub
                                huggingface_hub.snapshot_download = original_snapshot_download
                                if 'original_hf_hub_download' in locals():
                                    huggingface_hub.hf_hub_download = original_hf_hub_download
                                if 'original_cached_download' in locals():
                                    huggingface_hub.cached_download = original_cached_download
                            except:
                                pass
                            
                            try:
                                import requests
                                if 'original_requests_get' in locals():
                                    requests.get = original_requests_get
                                if 'original_requests_head' in locals():
                                    requests.head = original_requests_head
                            except:
                                pass
                            
                            try:
                                import httpx
                                if 'original_httpx_get' in locals():
                                    httpx.get = original_httpx_get
                                if 'original_httpx_head' in locals():
                                    httpx.head = original_httpx_head
                            except:
                                pass
                            
                            # Restaurer les variables d'environnement
                            if original_hf_home:
                                os.environ['HF_HOME'] = original_hf_home
                            elif 'HF_HOME' in os.environ:
                                del os.environ['HF_HOME']
                            if original_huggingface_hub_cache:
                                os.environ['HUGGINGFACE_HUB_CACHE'] = original_huggingface_hub_cache
                            elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
                                del os.environ['HUGGINGFACE_HUB_CACHE']
                            if 'HF_HUB_OFFLINE' in os.environ:
                                del os.environ['HF_HUB_OFFLINE']
                            if 'TRANSFORMERS_OFFLINE' in os.environ:
                                del os.environ['TRANSFORMERS_OFFLINE']
                    except Exception as e:
                        logger.error(f"❌ Failed to load model from local path: {e}")
                        self.pipeline = None
                        return
            else:
                # Essayer de charger depuis HuggingFace
                model_name = getattr(self.config, 'diarization_model', 'pyannote/speaker-diarization-3.1')
                
                if self.hf_token:
                    logger.info(f"📥 Downloading model '{model_name}' from HuggingFace with token...")
                    # pyannote.audio 3.x utilise 'token' au lieu de 'use_auth_token'
                    try:
                        self.pipeline = Pipeline.from_pretrained(
                            model_name,
                            token=self.hf_token
                        )
                    except TypeError:
                        # Fallback pour les anciennes versions
                        self.pipeline = Pipeline.from_pretrained(
                            model_name,
                            use_auth_token=self.hf_token
                        )
                else:
                    logger.warning(
                        "⚠️ No local model found and no Hugging Face token provided. "
                        "Diarization will be disabled. "
                        "Either place a model in /app/models/transcribe/pyannote-speaker-diarization "
                        "or set HF_TOKEN environment variable to enable."
                    )
                    self.pipeline = None
                    return
            
            # Déplacer le pipeline sur le bon device (CPU ou GPU)
            if torch is not None:
                use_gpu = getattr(self.config, 'diarization_use_gpu', True)
                
                if use_gpu and torch.cuda.is_available():
                    device = torch.device('cuda')
                    self.pipeline.to(device)
                    logger.info("✅ Diarization model loaded on GPU (optimized for speed)")
                else:
                    device = torch.device('cpu')
                    self.pipeline.to(device)
                    
                    # Limiter le nombre de threads PyTorch si configuré (pour réduire CPU)
                    num_threads = getattr(self.config, 'diarization_num_threads', 0)
                    if num_threads > 0:
                        torch.set_num_threads(num_threads)
                        logger.info(f"⚙️ Diarization: Limited PyTorch to {num_threads} thread(s) for CPU usage control")
                    else:
                        # Par défaut, limiter à la moitié des cores disponibles pour laisser de la marge
                        import os
                        try:
                            cpu_count = os.cpu_count() or 4
                            # Limiter à max 4 threads même si plus de cores disponibles
                            recommended_threads = min(cpu_count // 2, 4)
                            if recommended_threads > 0:
                                torch.set_num_threads(recommended_threads)
                                logger.info(f"⚙️ Diarization: Auto-limited PyTorch to {recommended_threads} thread(s) for CPU usage control")
                        except:
                            pass
                    
                    if use_gpu:
                        logger.info("⚠️ GPU requested but not available, using CPU")
                    else:
                        logger.info("✅ Diarization model loaded on CPU")
            else:
                logger.warning("⚠️ torch not available, model will use default device")
                
        except ImportError:
            logger.error(
                "❌ pyannote.audio not installed. "
                "Install it with: pip install pyannote.audio"
            )
            self.pipeline = None
        except Exception as e:
            logger.error(f"❌ Error loading diarization model: {e}")
            self.pipeline = None
    
    def diarize(self, audio_path: Path) -> List[Dict[str, float]]:
        """
        Effectue la diarisation sur un fichier audio avec optimisations.
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Liste de dictionnaires avec les segments de chaque locuteur:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        if self.pipeline is None:
            logger.warning("⚠️ Diarization pipeline not available, skipping diarization")
            return []
        
        try:
            from audio_utils import get_audio_duration
            
            audio_duration = get_audio_duration(audio_path)
            logger.info(f"🎤 Running diarization on {audio_path.name} (duration: {audio_duration:.1f}s)...")
            
            # Vérifier si on doit utiliser le traitement par chunks
            # Par défaut, utiliser 300s (5 min) pour réduire la charge CPU au lieu de 600s
            chunk_duration = getattr(self.config, 'diarization_chunk_duration_s', 300)
            
            if chunk_duration > 0 and audio_duration > chunk_duration:
                # Traitement par chunks pour les longs fichiers
                logger.info(f"⚡ Using chunk-based diarization (chunk size: {chunk_duration}s) to reduce CPU usage")
                return self._diarize_in_chunks(audio_path, audio_duration, chunk_duration)
            else:
                # Traitement normal pour les fichiers courts
                return self._diarize_single(audio_path)
            
        except Exception as e:
            logger.error(f"❌ Error during diarization: {e}", exc_info=True)
            return []
    
    def _diarize_single(self, audio_path: Path) -> List[Dict[str, float]]:
        """
        Effectue la diarisation sur un fichier audio entier.
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Liste de dictionnaires avec les segments de chaque locuteur
        """
        try:
            # Préparer les paramètres du pipeline
            diarization_params = {}
            
            # Ajouter min_speakers si configuré
            min_speakers = getattr(self.config, 'diarization_min_speakers', None)
            if min_speakers is not None:
                diarization_params['min_speakers'] = min_speakers
                logger.info(f"⚙️ Diarization: min_speakers={min_speakers}")
            
            # Ajouter max_speakers si configuré
            max_speakers = getattr(self.config, 'diarization_max_speakers', None)
            if max_speakers is not None:
                diarization_params['max_speakers'] = max_speakers
                logger.info(f"⚙️ Diarization: max_speakers={max_speakers}")
            
            # Exécuter la diarisation avec paramètres
            if diarization_params:
                # pyannote.audio 3.x accepte ces paramètres directement
                diarization = self.pipeline(str(audio_path), **diarization_params)
            else:
                diarization = self.pipeline(str(audio_path))
            
            # Extraire l'annotation depuis DiarizeOutput (pyannote.audio 3.1+)
            if not hasattr(diarization, 'speaker_diarization'):
                # Fallback vers exclusive_speaker_diarization si disponible
                if hasattr(diarization, 'exclusive_speaker_diarization'):
                    annotation = diarization.exclusive_speaker_diarization
                else:
                    raise AttributeError(
                        f"DiarizeOutput has neither 'speaker_diarization' nor 'exclusive_speaker_diarization'. "
                        f"Available attributes: {[attr for attr in dir(diarization) if not attr.startswith('_')]}"
                    )
            else:
                annotation = diarization.speaker_diarization
            
            # Convertir les segments en liste de dictionnaires
            segments = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker
                })
            
            # Compter le nombre de locuteurs uniques
            unique_speakers = set(seg["speaker"] for seg in segments)
            logger.info(
                f"✅ Diarization completed: {len(segments)} segments, "
                f"{len(unique_speakers)} speaker(s) detected"
            )
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Error during single diarization: {e}", exc_info=True)
            return []
    
    def _diarize_in_chunks(self, audio_path: Path, audio_duration: float, chunk_duration: float) -> List[Dict[str, float]]:
        """
        Effectue la diarisation sur un fichier audio en le découpant en chunks.
        Optimisé pour les longs fichiers audio (> 10 minutes).
        
        Args:
            audio_path: Chemin vers le fichier audio
            audio_duration: Durée totale de l'audio en secondes
            chunk_duration: Durée de chaque chunk en secondes
            
        Returns:
            Liste de dictionnaires avec les segments de chaque locuteur
        """
        try:
            from pydub import AudioSegment
            import tempfile
            from pathlib import Path as PathLib
            
            chunk_overlap = getattr(self.config, 'diarization_chunk_overlap_s', 5)
            min_speakers = getattr(self.config, 'diarization_min_speakers', None)
            max_speakers = getattr(self.config, 'diarization_max_speakers', None)
            
            logger.info(f"⚡ Processing {audio_duration:.1f}s audio in chunks of {chunk_duration}s (overlap: {chunk_overlap}s)")
            
            # Charger l'audio
            audio = AudioSegment.from_file(str(audio_path))
            total_segments = []
            temp_files = []
            
            # Traiter par chunks avec chevauchement
            num_chunks = int((audio_duration + chunk_overlap) / (chunk_duration - chunk_overlap)) + 1
            
            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * (chunk_duration - chunk_overlap)
                end_time = min(start_time + chunk_duration, audio_duration)
                
                if start_time >= audio_duration:
                    break
                
                logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Extraire le chunk
                chunk_start_ms = int(start_time * 1000)
                chunk_end_ms = int(end_time * 1000)
                chunk_audio = audio[chunk_start_ms:chunk_end_ms]
                
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = PathLib(tmp_file.name)
                    chunk_audio.export(str(tmp_path), format='wav')
                    temp_files.append(tmp_path)
                
                try:
                    # Diariser le chunk
                    diarization_params = {}
                    if min_speakers is not None:
                        diarization_params['min_speakers'] = min_speakers
                    if max_speakers is not None:
                        diarization_params['max_speakers'] = max_speakers
                    
                    if diarization_params:
                        diarization = self.pipeline(str(tmp_path), **diarization_params)
                    else:
                        diarization = self.pipeline(str(tmp_path))
                    
                    # Extraire l'annotation
                    if hasattr(diarization, 'speaker_diarization'):
                        annotation = diarization.speaker_diarization
                    elif hasattr(diarization, 'exclusive_speaker_diarization'):
                        annotation = diarization.exclusive_speaker_diarization
                    else:
                        logger.warning(f"⚠️ Chunk {chunk_idx + 1}: Could not extract annotation")
                        continue
                    
                    # Convertir les segments et ajuster les timestamps
                    chunk_offset = start_time
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        total_segments.append({
                            "start": round(turn.start + chunk_offset, 2),
                            "end": round(turn.end + chunk_offset, 2),
                            "speaker": speaker
                        })
                    
                    logger.info(f"✅ Chunk {chunk_idx + 1}/{num_chunks}: {len([s for s in total_segments if chunk_offset <= s['start'] < chunk_offset + chunk_duration])} segments")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {chunk_idx + 1}: {e}", exc_info=True)
                    continue
                finally:
                    # Nettoyer le fichier temporaire
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except:
                        pass
            
            # Fusionner les segments qui se chevauchent dans la zone d'overlap
            total_segments = self._merge_overlapping_segments(total_segments, chunk_overlap)
            
            # Trier par timestamp
            total_segments.sort(key=lambda x: x['start'])
            
            # Compter le nombre de locuteurs uniques
            unique_speakers = set(seg["speaker"] for seg in total_segments)
            logger.info(
                f"✅ Chunk-based diarization completed: {len(total_segments)} segments, "
                f"{len(unique_speakers)} speaker(s) detected"
            )
            
            return total_segments
            
        except Exception as e:
            logger.error(f"❌ Error during chunk-based diarization: {e}", exc_info=True)
            return []
    
    def _merge_overlapping_segments(self, segments: List[Dict], overlap_duration: float) -> List[Dict]:
        """
        Fusionne les segments qui se chevauchent dans la zone d'overlap entre chunks.
        Prend le locuteur avec le plus grand chevauchement.
        
        Args:
            segments: Liste de segments
            overlap_duration: Durée de chevauchement en secondes
            
        Returns:
            Liste de segments fusionnés
        """
        if not segments:
            return segments
        
        # Trier par timestamp
        segments = sorted(segments, key=lambda x: (x['start'], x['end']))
        merged = []
        
        for current in segments:
            if not merged:
                merged.append(current)
                continue
            
            previous = merged[-1]
            
            # Vérifier si les segments se chevauchent dans la zone d'overlap
            overlap_start = max(previous['start'], current['start'])
            overlap_end = min(previous['end'], current['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            # Si le chevauchement est dans la zone d'overlap prévue et significatif
            if overlap > 0 and overlap <= overlap_duration + 1.0:  # Tolérance de 1s
                # Prendre le locuteur avec le plus grand chevauchement
                prev_overlap = min(previous['end'], current['end']) - max(previous['start'], current['start'])
                if prev_overlap > overlap / 2:
                    # Le segment précédent a plus de poids
                    merged[-1] = previous
                    continue
                else:
                    # Le segment courant a plus de poids, remplacer
                    merged[-1] = current
                    continue
            
            # Pas de chevauchement significatif, ajouter comme nouveau segment
            merged.append(current)
        
        return merged
    
    def assign_speakers_to_segments(
        self, 
        transcription_segments: List[Dict], 
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Assigne les locuteurs aux segments de transcription en fonction des timestamps.
        
        Args:
            transcription_segments: Segments de transcription avec start/end/text
            diarization_segments: Segments de diarisation avec start/end/speaker
            
        Returns:
            Segments de transcription avec le champ 'speaker' ajouté
        """
        if not diarization_segments:
            logger.warning("⚠️ No diarization segments, skipping speaker assignment")
            return transcription_segments
        
        # Créer une liste des segments avec speakers assignés
        segments_with_speakers = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_mid = (trans_start + trans_end) / 2.0
            
            # Trouver le locuteur qui parle au milieu du segment de transcription
            speaker = None
            max_overlap = 0.0
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg["start"]
                diar_end = diar_seg["end"]
                
                # Calculer l'overlap entre le segment de transcription et le segment de diarisation
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                # Si le milieu du segment de transcription est dans ce segment de diarisation
                if diar_start <= trans_mid <= diar_end:
                    speaker = diar_seg["speaker"]
                    break
                
                # Sinon, garder le segment avec le plus d'overlap
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = diar_seg["speaker"]
            
            # Si aucun locuteur trouvé, utiliser "UNKNOWN"
            if speaker is None:
                speaker = "UNKNOWN"
            
            # Créer le segment avec le speaker
            seg_with_speaker = trans_seg.copy()
            seg_with_speaker["speaker"] = speaker
            segments_with_speakers.append(seg_with_speaker)
        
        logger.info(
            f"✅ Assigned speakers to {len(segments_with_speakers)} transcription segments"
        )
        
        return segments_with_speakers

