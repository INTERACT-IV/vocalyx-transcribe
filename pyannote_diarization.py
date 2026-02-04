"""
vocalyx-transcribe/pyannote_diarization.py
Service de diarisation basé sur pyannote.audio (comme WhisperX)

Cette solution utilise un modèle ML pour la diarisation des fichiers audio MONO.
Même implémentation que WhisperX pour garantir la compatibilité.

AVANTAGES:
- Fonctionne sur audio mono (contrairement à la diarisation stéréo)
- Plus précis grâce au modèle ML
- Compatible avec l'implémentation WhisperX

INCONVÉNIENTS:
- Plus lent que la diarisation stéréo (modèle ML)
- Nécessite un modèle pyannote (téléchargé automatiquement)
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch

# Configurer le cache HuggingFace AVANT l'import de pyannote
# Les modèles pyannote seront dans /app/models/pyannote/ (même niveau que Whisper)
pyannote_cache_base = Path("/app/models/pyannote")
pyannote_cache_base.mkdir(parents=True, exist_ok=True)

# HF_HOME doit pointer vers le répertoire qui contiendra le dossier 'hub/'
os.environ['HF_HOME'] = str(pyannote_cache_base)
os.environ['TRANSFORMERS_CACHE'] = str(pyannote_cache_base / 'transformers')
os.environ['HF_DATASETS_CACHE'] = str(pyannote_cache_base / 'datasets')

# S'assurer que le répertoire hub existe
hub_dir = pyannote_cache_base / 'hub'
hub_dir.mkdir(parents=True, exist_ok=True)

# Note: Les variables d'environnement HF_HOME, TRANSFORMERS_CACHE et HF_DATASETS_CACHE
# sont suffisantes pour configurer le cache HuggingFace. Pas besoin d'accéder directement
# à huggingface_hub.constants qui peut ne pas exister dans toutes les versions.

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

import soundfile as sf

logger = logging.getLogger("vocalyx")

# Sample rate attendu par pyannote (identique à WhisperX)
SAMPLE_RATE = 16000


def load_audio(audio_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Charge un fichier audio et le convertit en numpy array.
    Même fonction que WhisperX pour garantir la compatibilité.
    
    IMPORTANT: Pas de normalisation de volume (peak normalization) pour éviter la saturation.
    WhisperX ne fait PAS de normalisation de volume, seulement conversion int16 -> float32.
    La normalisation de volume peut causer des problèmes avec pyannote si le son est déjà fort.
    
    Args:
        audio_path: Chemin vers le fichier audio
        sample_rate: Sample rate cible (défaut: 16000)
        
    Returns:
        numpy.ndarray: Audio en mono, en float32, valeurs entre -1 et 1 (sans normalisation de volume)
    """
    try:
        # Essayer d'abord avec ffmpeg (comme WhisperX) si disponible
        try:
            import subprocess
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads", "0",
                "-i", str(audio_path),
                "-f", "s16le",
                "-ac", "1",  # Mono
                "-acodec", "pcm_s16le",
                "-ar", str(sample_rate),
                "-",
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
            # Convertir int16 -> float32 comme WhisperX (division par 32768.0)
            audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            logger.debug("✅ Loaded audio with ffmpeg (like WhisperX)")
            return audio
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: utiliser soundfile si ffmpeg n'est pas disponible
            logger.debug("⚠️ ffmpeg not available, using soundfile as fallback")
            pass
        
        # Fallback avec soundfile (compatible avec WhisperX)
        # IMPORTANT: Ne PAS utiliser dtype=np.float32 directement car cela peut normaliser automatiquement
        # On lit en int16 puis on convertit manuellement comme WhisperX
        try:
            # Essayer de lire en int16 d'abord (comme WhisperX)
            audio, sr = sf.read(str(audio_path), dtype=np.int16)
            # Convertir en mono si stéréo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            # Convertir int16 -> float32 comme WhisperX
            audio = audio.astype(np.float32) / 32768.0
        except (ValueError, TypeError):
            # Si le fichier n'est pas en int16, lire directement en float32
            audio, sr = sf.read(str(audio_path), dtype=np.float32)
            # Convertir en mono si stéréo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            # Ne PAS normaliser - garder les valeurs telles quelles (comme WhisperX)
            # soundfile.read() avec dtype=np.float32 retourne déjà des valeurs entre -1 et 1
        
        # Resample si nécessaire (comme WhisperX)
        if sr != sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)
        
        # IMPORTANT: Ne PAS faire de normalisation de volume (peak normalization)
        # WhisperX ne le fait pas, et cela peut causer de la saturation si le son est déjà fort
        
        # Vérifier que les valeurs sont dans une plage raisonnable (pas de saturation)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            logger.warning(
                f"⚠️ Audio values exceed [-1, 1] range (max: {max_val:.3f}). "
                f"This might indicate saturation. Clamping to [-1, 1]."
            )
            audio = np.clip(audio, -1.0, 1.0)
        elif max_val > 0.95:
            logger.debug(
                f"🔍 Audio is close to saturation (max: {max_val:.3f})"
            )
        
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"❌ Error loading audio {audio_path}: {e}", exc_info=True)
        raise


class PyannoteDiarizationService:
    """
    Service de diarisation basé sur pyannote.audio (comme WhisperX).
    
    Utilisé pour les fichiers audio MONO.
    Même implémentation que WhisperX pour garantir la compatibilité.
    """
    
    def __init__(self, config=None):
        """
        Initialise le service de diarisation pyannote.
        
        Args:
            config: Configuration avec les paramètres pyannote
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Install it with: pip install pyannote.audio"
            )
        
        self.config = config
        
        # Récupérer les paramètres de configuration
        device_str = getattr(config, 'device', 'cpu')
        if isinstance(device_str, str):
            device = torch.device(device_str)
        else:
            device = device_str
        
        model_name = getattr(
            config, 
            'pyannote_model', 
            'pyannote/speaker-diarization-3.1'
        )
        
        # Le cache est déjà configuré au niveau du module (avant l'import de pyannote)
        # Utiliser le répertoire de cache configuré globalement
        pyannote_cache_dir = pyannote_cache_base
        hub_dir = pyannote_cache_dir / 'hub'
        
        # Vérifier si le modèle existe déjà localement
        # Les modèles HuggingFace sont dans HF_HOME/hub/models--pyannote--speaker-diarization-3.1/
        model_cache_name = model_name.replace("/", "--")
        model_cache_path = hub_dir / f'models--{model_cache_name}'
        model_exists_locally = model_cache_path.exists() and any(model_cache_path.iterdir())
        
        logger.info(f"📁 Pyannote cache directory: {pyannote_cache_dir}")
        logger.info(f"📁 Hub directory: {hub_dir}")
        logger.info(f"📁 Model cache path: {model_cache_path}")
        
        if model_exists_locally:
            logger.info(f"✅ Pyannote model found locally at {model_cache_path}")
        else:
            logger.info(f"📥 Pyannote model will be downloaded to {pyannote_cache_dir}")
        
        self.pyannote_cache_dir = pyannote_cache_dir
        self.model_exists_locally = model_exists_locally
        
        # Récupérer le token : config > variable d'environnement > None
        # Les versions récentes de pyannote.audio utilisent 'token' au lieu de 'use_auth_token'
        # et supportent aussi les variables d'environnement HF_TOKEN ou HUGGING_FACE_HUB_TOKEN
        auth_token = getattr(config, 'pyannote_auth_token', None)
        
        # Nettoyer le token s'il existe (enlever les espaces, etc.)
        if auth_token:
            auth_token = str(auth_token).strip()
            if not auth_token:
                auth_token = None
        
        if not auth_token:
            # Essayer les variables d'environnement standard de HuggingFace
            auth_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            if auth_token:
                auth_token = str(auth_token).strip()
                if not auth_token:
                    auth_token = None
        
        # Log pour debug (masquer le token pour la sécurité)
        if auth_token:
            token_preview = auth_token[:8] + "..." if len(auth_token) > 8 else "***"
            logger.info(f"🔑 Token found: {token_preview} (length: {len(auth_token)})")
        else:
            logger.info("ℹ️ No token found in config or environment variables")
        
        # Paramètres optionnels pour la diarisation
        self.num_speakers = getattr(config, 'pyannote_num_speakers', None)
        self.min_speakers = getattr(config, 'pyannote_min_speakers', None)
        self.max_speakers = getattr(config, 'pyannote_max_speakers', None)
        
        logger.info(f"🎯 Loading pyannote diarization model: {model_name}")
        logger.info(f"📊 Device: {device}")
        logger.info(f"📁 Cache directory: {self.pyannote_cache_dir}")
        
        try:
            # Préparer les arguments pour Pipeline.from_pretrained
            pipeline_kwargs = {}
            
            # Ajouter le token si disponible
            if auth_token:
                logger.info("🔑 Using HuggingFace token for authentication")
                pipeline_kwargs['token'] = auth_token
            else:
                logger.info("ℹ️ No token provided, pyannote will use environment variables if available")
            
            # Essayer d'utiliser local_files_only si le modèle existe déjà
            # Note: local_files_only n'est peut-être pas supporté par toutes les versions de pyannote
            if self.model_exists_locally:
                # Essayer avec local_files_only pour éviter les téléchargements
                pipeline_kwargs['local_files_only'] = True
                logger.info("🔍 Trying to load model with local_files_only=True")
            
            # Essayer d'abord avec 'token' (versions récentes)
            # Essayer aussi avec cache_dir si supporté
            try:
                # Essayer avec cache_dir explicite (si supporté)
                try:
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        cache_dir=str(self.pyannote_cache_dir),
                        **pipeline_kwargs
                    ).to(device)
                except TypeError:
                    # Si cache_dir n'est pas supporté, utiliser sans
                    logger.debug("⚠️ cache_dir parameter not supported, using environment variables")
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        **pipeline_kwargs
                    ).to(device)
            except (OSError, FileNotFoundError) as e:
                # Si local_files_only=True et modèle non trouvé, réessayer sans
                if self.model_exists_locally and pipeline_kwargs.get('local_files_only'):
                    logger.warning(f"⚠️ Model not found locally despite cache check, retrying without local_files_only: {e}")
                    pipeline_kwargs.pop('local_files_only', None)
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        **pipeline_kwargs
                    ).to(device)
                else:
                    raise
            except TypeError as e:
                # Si 'token' n'est pas supporté, essayer 'use_auth_token' (anciennes versions)
                if "unexpected keyword argument 'token'" in str(e):
                    logger.info("🔄 Trying with 'use_auth_token' parameter (older pyannote.audio version)")
                    if auth_token:
                        pipeline_kwargs['use_auth_token'] = pipeline_kwargs.pop('token', None)
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        **pipeline_kwargs
                    ).to(device)
                elif "unexpected keyword argument 'local_files_only'" in str(e):
                    # Si local_files_only n'est pas supporté, réessayer sans
                    pipeline_kwargs.pop('local_files_only', None)
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        **pipeline_kwargs
                    ).to(device)
                elif "unexpected keyword argument 'cache_dir'" in str(e):
                    # Si cache_dir n'est pas supporté, réessayer sans
                    logger.debug("⚠️ cache_dir parameter not supported in this pyannote version")
                    # Réessayer sans cache_dir mais avec les variables d'environnement
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        **pipeline_kwargs
                    ).to(device)
                else:
                    raise
            logger.info("✅ Pyannote diarization service initialized and ready")
        except Exception as e:
            logger.error(f"❌ Failed to load pyannote model: {e}", exc_info=True)
            raise
    
    def diarize(self, audio_path: Path) -> List[Dict[str, float]]:
        """
        Effectue la diarisation pyannote sur un fichier audio.
        Même format de retour que StereoDiarizationService pour compatibilité.
        
        Args:
            audio_path: Chemin vers le fichier audio (mono ou stéréo)
            
        Returns:
            Liste de dictionnaires avec les segments de chaque locuteur:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        try:
            logger.info(f"🎤 Running pyannote diarization on {audio_path.name}...")
            
            # Charger l'audio (même méthode que WhisperX)
            audio = load_audio(audio_path, SAMPLE_RATE)
            duration = len(audio) / SAMPLE_RATE
            logger.info(f"📏 Audio duration: {duration:.1f}s")
            
            # Préparer les données pour pyannote (même format que WhisperX)
            audio_data = {
                'waveform': torch.from_numpy(audio[None, :]),  # Ajouter dimension batch
                'sample_rate': SAMPLE_RATE
            }
            
            # Exécuter la diarisation avec les paramètres optionnels
            # Log des paramètres utilisés pour debug
            diarization_params = {}
            if self.num_speakers is not None:
                diarization_params['num_speakers'] = self.num_speakers
                logger.info(f"🎯 Using num_speakers={self.num_speakers} (forcing exact speaker count)")
            if self.min_speakers is not None:
                diarization_params['min_speakers'] = self.min_speakers
                logger.info(f"🎯 Using min_speakers={self.min_speakers}")
            if self.max_speakers is not None:
                diarization_params['max_speakers'] = self.max_speakers
                logger.info(f"🎯 Using max_speakers={self.max_speakers}")
            
            diarization_result = self.model(
                audio_data,
                **diarization_params
            )
            
            # Gérer différents formats de retour selon la version de pyannote.audio
            # Les versions récentes peuvent retourner DiarizeOutput au lieu d'Annotation directement
            diarization = None
            
            # Essayer différentes méthodes pour accéder à l'Annotation
            # 1. Vérifier si c'est directement une Annotation (format classique)
            if hasattr(diarization_result, 'itertracks'):
                diarization = diarization_result
                logger.debug("✅ Using direct Annotation format")
            # 2. Essayer d'accéder directement aux attributs communs de DiarizeOutput
            # Les versions récentes de pyannote retournent DiarizeOutput avec speaker_diarization ou exclusive_speaker_diarization
            else:
                # Essayer d'abord les attributs les plus courants
                # PRIORITÉ: speaker_diarization (avec chevauchements) > exclusive_speaker_diarization (sans chevauchements)
                # speaker_diarization est préféré car il conserve tous les segments même en cas de chevauchement,
                # ce qui permet une meilleure détection des speakers qui parlent en même temps ou rapidement
                for attr_name in ['speaker_diarization', 'exclusive_speaker_diarization', 'annotation']:
                    try:
                        attr_value = getattr(diarization_result, attr_name, None)
                        if attr_value is not None and hasattr(attr_value, 'itertracks'):
                            diarization = attr_value
                            logger.debug(f"✅ Found Annotation via '{attr_name}' attribute")
                            if attr_name == 'speaker_diarization':
                                logger.info("✅ Using speaker_diarization (with overlaps) - better for detecting all speakers")
                            elif attr_name == 'exclusive_speaker_diarization':
                                logger.info("⚠️ Using exclusive_speaker_diarization (without overlaps) - may miss overlapping speakers")
                            break
                    except (AttributeError, TypeError):
                        continue
                
                # Si toujours None, essayer l'accès direct à .annotation (peut lever AttributeError)
                if diarization is None:
                    try:
                        diarization = diarization_result.annotation
                        if hasattr(diarization, 'itertracks'):
                            logger.debug("✅ Using DiarizeOutput.annotation (direct access)")
                    except AttributeError:
                        pass
            
            # Si toujours None, logger pour debug et essayer toutes les méthodes possibles
            if diarization is None:
                logger.debug(f"🔍 Could not find Annotation directly, searching in DiarizeOutput attributes...")
                all_attrs = [a for a in dir(diarization_result) if not a.startswith('_')]
                logger.debug(f"🔍 Available attributes: {all_attrs}")
                
                # Essayer d'accéder aux attributs qui contiennent l'annotation
                # Priorité: speaker_diarization (standard avec chevauchements) > exclusive_speaker_diarization (sans chevauchements)
                priority_attrs = ['speaker_diarization', 'exclusive_speaker_diarization', 'annotation', 'diarization']
                
                for attr_name in priority_attrs:
                    if attr_name in all_attrs:
                        try:
                            attr_value = getattr(diarization_result, attr_name)
                            logger.debug(f"🔍 Checking attribute '{attr_name}': type={type(attr_value)}, has_itertracks={hasattr(attr_value, 'itertracks') if attr_value is not None else False}")
                            if attr_value is not None and hasattr(attr_value, 'itertracks'):
                                diarization = attr_value
                                logger.info(f"✅ Found Annotation in attribute '{attr_name}'")
                                break
                        except Exception as e:
                            logger.debug(f"⚠️ Could not access attribute '{attr_name}': {e}")
                            continue
                
                # Si pas trouvé dans les attributs prioritaires, essayer tous les autres
                if diarization is None:
                    for attr_name in all_attrs:
                        if attr_name not in priority_attrs:
                            try:
                                attr_value = getattr(diarization_result, attr_name)
                                if attr_value is not None and hasattr(attr_value, 'itertracks'):
                                    diarization = attr_value
                                    logger.info(f"✅ Found Annotation in attribute '{attr_name}'")
                                    break
                            except Exception as e:
                                logger.debug(f"⚠️ Could not access attribute '{attr_name}': {e}")
                                continue
                
                # Essayer aussi l'accès par index si c'est un tuple ou NamedTuple
                if diarization is None:
                    try:
                        if hasattr(diarization_result, '_fields'):  # NamedTuple
                            logger.info(f"🔍 DiarizeOutput appears to be a NamedTuple with fields: {diarization_result._fields}")
                            # Essayer d'accéder au premier champ (généralement l'annotation)
                            if len(diarization_result._fields) > 0:
                                first_field = diarization_result._fields[0]
                                first_value = getattr(diarization_result, first_field)
                                logger.info(f"🔍 First field '{first_field}' type: {type(first_value)}")
                                if hasattr(first_value, 'itertracks'):
                                    diarization = first_value
                                    logger.info(f"✅ Found Annotation in NamedTuple field '{first_field}'")
                        elif hasattr(diarization_result, '__getitem__'):
                            # Essayer l'accès par index
                            try:
                                indexed_value = diarization_result[0]
                                if hasattr(indexed_value, 'itertracks'):
                                    diarization = indexed_value
                                    logger.info("✅ Found Annotation via index [0]")
                            except (IndexError, TypeError):
                                pass
                    except Exception as e:
                        logger.debug(f"⚠️ Error trying NamedTuple/index access: {e}")
                
                # Si toujours None, essayer d'utiliser directement le résultat si c'est itérable
                if diarization is None:
                    # Vérifier si DiarizeOutput peut être converti directement
                    try:
                        # Essayer de convertir en dict ou d'accéder comme un dict
                        if hasattr(diarization_result, '_asdict'):
                            # NamedTuple avec _asdict
                            result_dict = diarization_result._asdict()
                            logger.info(f"🔍 Converted to dict: {list(result_dict.keys())}")
                            # Chercher 'annotation' dans le dict
                            if 'annotation' in result_dict:
                                diarization = result_dict['annotation']
                                logger.info("✅ Found annotation in _asdict()")
                        elif isinstance(diarization_result, (list, tuple)) and len(diarization_result) > 0:
                            # C'est peut-être directement un tuple/list avec l'annotation en premier
                            first_item = diarization_result[0]
                            if hasattr(first_item, 'itertracks'):
                                diarization = first_item
                                logger.info("✅ Found Annotation as first item in tuple/list")
                    except Exception as e:
                        logger.debug(f"⚠️ Error in conversion attempts: {e}")
                    
                    # Dernier recours : essayer d'itérer directement
                    if diarization is None and hasattr(diarization_result, '__iter__') and not isinstance(diarization_result, (str, bytes)):
                        diarization = diarization_result
            
            # Convertir au format attendu (même que StereoDiarizationService)
            diarization_segments = []
            
            if diarization is None:
                # Si diarization est None, essayer de traiter directement diarization_result
                logger.debug("⚠️ diarization is None, trying to process diarization_result directly")
                
                # Essayer d'itérer directement si c'est itérable
                try:
                    if isinstance(diarization_result, (list, tuple)):
                        # C'est une liste de segments
                        for item in diarization_result:
                            if isinstance(item, dict):
                                diarization_segments.append({
                                    "start": round(item.get('start', 0), 2),
                                    "end": round(item.get('end', 0), 2),
                                    "speaker": item.get('speaker', item.get('label', 'UNKNOWN'))
                                })
                            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                                segment, track, label = item[0], item[1], item[2]
                                if hasattr(segment, 'start'):
                                    diarization_segments.append({
                                        "start": round(segment.start, 2),
                                        "end": round(segment.end, 2),
                                        "speaker": label
                                    })
                                else:
                                    diarization_segments.append({
                                        "start": round(float(segment), 2),
                                        "end": round(float(track), 2),
                                        "speaker": str(label)
                                    })
                    elif hasattr(diarization_result, '__iter__'):
                        # C'est un itérable mais pas une liste
                        for item in diarization_result:
                            if isinstance(item, (tuple, list)) and len(item) >= 3:
                                segment, track, label = item[0], item[1], item[2]
                                if hasattr(segment, 'start'):
                                    diarization_segments.append({
                                        "start": round(segment.start, 2),
                                        "end": round(segment.end, 2),
                                        "speaker": label
                                    })
                except Exception as e:
                    logger.error(f"❌ Error processing diarization_result directly: {e}")
                    logger.error(f"❌ Result type: {type(diarization_result)}")
                    return []
            elif hasattr(diarization, 'itertracks'):
                # Format Annotation standard (comme WhisperX)
                for segment, track, label in diarization.itertracks(yield_label=True):
                    diarization_segments.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "speaker": label  # Format: "SPEAKER_00", "SPEAKER_01", etc.
                    })
            elif diarization_result is not None and hasattr(diarization_result, '__dict__'):
                # Essayer d'accéder aux attributs de l'objet
                logger.warning(f"⚠️ Trying to access attributes of {type(diarization_result)}")
                attrs = vars(diarization_result)
                logger.debug(f"🔍 Object attributes: {list(attrs.keys())}")
                
                # Chercher un attribut qui pourrait contenir l'annotation
                for attr_name in ['annotation', 'diarization', 'segments', 'tracks', 'result']:
                    if hasattr(diarization_result, attr_name):
                        attr_value = getattr(diarization_result, attr_name)
                        logger.debug(f"🔍 Found attribute '{attr_name}': {type(attr_value)}")
                        if hasattr(attr_value, 'itertracks'):
                            diarization = attr_value
                            logger.info(f"✅ Found Annotation in attribute '{attr_name}'")
                            break
                        elif isinstance(attr_value, list) and len(attr_value) > 0:
                            # C'est peut-être une liste de segments
                            diarization = None
                            diarization_result = attr_value
                            logger.info(f"✅ Found list in attribute '{attr_name}', treating as segments")
                            break
                
                if diarization is None and diarization_result is not None:
                    # Essayer d'accéder comme un dictionnaire avec des clés communes
                    if hasattr(diarization_result, 'get') or isinstance(diarization_result, dict):
                        # C'est peut-être un dictionnaire-like
                        segments_data = None
                        if isinstance(diarization_result, dict):
                            segments_data = diarization_result.get('segments', diarization_result.get('tracks', diarization_result.get('diarization', [])))
                        elif hasattr(diarization_result, 'get'):
                            segments_data = diarization_result.get('segments', diarization_result.get('tracks', []))
                        
                        if segments_data:
                            for seg_data in segments_data:
                                if isinstance(seg_data, dict):
                                    diarization_segments.append({
                                        "start": round(seg_data.get('start', 0), 2),
                                        "end": round(seg_data.get('end', 0), 2),
                                        "speaker": seg_data.get('speaker', seg_data.get('label', 'UNKNOWN'))
                                    })
                                elif isinstance(seg_data, (list, tuple)) and len(seg_data) >= 2:
                                    diarization_segments.append({
                                        "start": round(seg_data[0], 2),
                                        "end": round(seg_data[1], 2),
                                        "speaker": seg_data[2] if len(seg_data) > 2 else 'UNKNOWN'
                                    })
                    else:
                        logger.error(f"❌ Cannot extract segments from diarization result type: {type(diarization_result)}")
                        logger.error(f"❌ Available methods: {[m for m in dir(diarization_result) if not m.startswith('_')]}")
                        return []
            else:
                logger.error(f"❌ Cannot extract segments from diarization result type: {type(diarization_result)}")
                logger.error(f"❌ Result: {diarization_result}")
                return []
            
            # Trier par timestamp de début
            diarization_segments.sort(key=lambda x: x['start'])
            
            # Compter le nombre de locuteurs uniques
            unique_speakers = set(seg["speaker"] for seg in diarization_segments)
            logger.info(
                f"✅ Pyannote diarization completed: {len(diarization_segments)} segments, "
                f"{len(unique_speakers)} speaker(s) detected: {sorted(unique_speakers)}"
            )
            
            # Logger quelques exemples de segments pour vérifier les speakers
            if len(unique_speakers) > 1:
                # Logger un exemple de chaque speaker
                for speaker_id in sorted(unique_speakers):
                    example_seg = next((s for s in diarization_segments if s["speaker"] == speaker_id), None)
                    if example_seg:
                        logger.debug(f"🔍 Example segment for {speaker_id}: {example_seg['start']:.2f}s - {example_seg['end']:.2f}s")
            
            # Logger la distribution des speakers
            speaker_distribution = {}
            for seg in diarization_segments:
                speaker = seg["speaker"]
                speaker_distribution[speaker] = speaker_distribution.get(speaker, 0) + 1
            logger.debug(f"🔍 Speaker distribution in diarization: {speaker_distribution}")
            
            return diarization_segments
            
        except Exception as e:
            logger.error(f"❌ Error during pyannote diarization: {e}", exc_info=True)
            return []
    
    def assign_speakers_to_segments(
        self, 
        transcription_segments: List[Dict], 
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Assigne les locuteurs aux segments de transcription en fonction des timestamps.
        Même implémentation que StereoDiarizationService pour compatibilité.
        
        Args:
            transcription_segments: Segments de transcription avec start/end/text
            diarization_segments: Segments de diarisation avec start/end/speaker
            
        Returns:
            Segments de transcription avec le champ 'speaker' ajouté
        """
        if not diarization_segments:
            logger.warning("⚠️ No diarization segments, skipping speaker assignment")
            return transcription_segments
        
        # Logger les segments de diarisation pour debug
        unique_diarization_speakers = set(seg["speaker"] for seg in diarization_segments)
        logger.info(f"🔍 Diarization segments: {len(diarization_segments)} segments with speakers: {unique_diarization_speakers}")
        
        # Logger quelques exemples de segments de diarisation pour comprendre leur répartition
        logger.info(f"🔍 First few diarization segments: {diarization_segments[:5]}")
        logger.info(f"🔍 Last few diarization segments: {diarization_segments[-5:]}")
        
        # Logger la répartition temporelle par speaker
        for speaker_id in sorted(unique_diarization_speakers):
            speaker_segments = [s for s in diarization_segments if s["speaker"] == speaker_id]
            if speaker_segments:
                first_seg = speaker_segments[0]
                last_seg = speaker_segments[-1]
                total_duration = sum(s["end"] - s["start"] for s in speaker_segments)
                logger.info(
                    f"🔍 {speaker_id}: {len(speaker_segments)} segments, "
                    f"first: [{first_seg['start']:.2f}-{first_seg['end']:.2f}], "
                    f"last: [{last_seg['start']:.2f}-{last_seg['end']:.2f}], "
                    f"total duration: {total_duration:.2f}s"
                )
        
        # Logger les segments de transcription pour comparaison
        logger.info(f"🔍 Transcription segments: {len(transcription_segments)} segments")
        if transcription_segments:
            logger.info(f"🔍 First transcription segment: {transcription_segments[0]}")
            logger.info(f"🔍 Last transcription segment: {transcription_segments[-1]}")
        
        # Créer une liste des segments avec speakers assignés
        segments_with_speakers = []
        speaker_assignment_count = {}
        
        # AMÉLIORATION : Détecter les changements de speaker basés sur les patterns temporels
        # Si un speaker apparaît seulement après un certain temps, vérifier s'il devrait apparaître plus tôt
        # en analysant les segments de transcription qui précèdent sa première détection
        
        # Trouver le premier segment de diarisation pour chaque speaker
        first_segment_by_speaker = {}
        for speaker_id in unique_diarization_speakers:
            speaker_segments = [s for s in diarization_segments if s["speaker"] == speaker_id]
            if speaker_segments:
                first_segment_by_speaker[speaker_id] = min(speaker_segments, key=lambda x: x['start'])
        
        # Si un speaker n'apparaît qu'après plusieurs secondes, vérifier s'il devrait être détecté plus tôt
        # en analysant les segments de transcription qui précèdent sa première détection
        early_segments_candidates = {}
        if len(first_segment_by_speaker) > 1:
            # Trouver le speaker qui apparaît le plus tard
            latest_speaker = max(first_segment_by_speaker.items(), key=lambda x: x[1]['start'])
            latest_speaker_id, latest_first_seg = latest_speaker
            
            # Si ce speaker apparaît après 5 secondes, analyser les segments précédents
            if latest_first_seg['start'] > 5.0:
                logger.info(
                    f"🔍 Speaker {latest_speaker_id} detected late (first at {latest_first_seg['start']:.2f}s). "
                    f"Analyzing early transcription segments for potential early detection..."
                )
                
                # Analyser les premiers segments de transcription (avant la première détection de ce speaker)
                early_trans_segments = [
                    seg for seg in transcription_segments 
                    if seg['start'] < latest_first_seg['start']
                ]
                
                if early_trans_segments:
                    logger.info(
                        f"🔍 Found {len(early_trans_segments)} transcription segments before "
                        f"{latest_speaker_id} first detection at {latest_first_seg['start']:.2f}s"
                    )
        
        # Utiliser la même logique que WhisperX : sommer les intersections par speaker
        # C'est plus simple et plus efficace que de chercher le meilleur segment individuel
        for idx, trans_seg in enumerate(transcription_segments):
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            
            # Calculer l'intersection avec chaque segment de diarisation et grouper par speaker
            speaker_intersections = {}
            
            # Logger tous les segments de diarisation qui chevauchent ce segment de transcription
            overlapping_diar_segments = []
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg["start"]
                diar_end = diar_seg["end"]
                speaker = diar_seg["speaker"]
                
                # Calculer l'intersection (overlap) entre le segment de transcription et le segment de diarisation
                intersection_start = max(trans_start, diar_start)
                intersection_end = min(trans_end, diar_end)
                intersection = max(0.0, intersection_end - intersection_start)
                
                # Sommer les intersections par speaker (comme WhisperX)
                if intersection > 0:
                    if speaker not in speaker_intersections:
                        speaker_intersections[speaker] = 0.0
                    speaker_intersections[speaker] += intersection
                    overlapping_diar_segments.append({
                        'speaker': speaker,
                        'diar_start': diar_start,
                        'diar_end': diar_end,
                        'intersection': intersection
                    })
            
            # Logger pour les premiers et derniers segments de transcription
            if idx < 2 or idx >= len(transcription_segments) - 2:
                logger.info(
                    f"🔍 Trans seg {idx} [{trans_start:.2f}-{trans_end:.2f}]: "
                    f"intersections by speaker: {speaker_intersections}, "
                    f"overlapping diar segments: {len(overlapping_diar_segments)}"
                )
                if overlapping_diar_segments:
                    logger.info(f"🔍   Overlapping diarization details (first 3): {overlapping_diar_segments[:3]}")
            
            # Logger pour TOUS les segments de transcription qui ont plusieurs speakers
            if len(speaker_intersections) > 1:
                logger.info(
                    f"🔍 Trans seg {idx} [{trans_start:.2f}-{trans_end:.2f}]: "
                    f"MULTIPLE SPEAKERS detected! intersections: {speaker_intersections}"
                )
                logger.info(f"🔍   Overlapping diarization segments: {overlapping_diar_segments}")
            elif len(speaker_intersections) == 0:
                logger.warning(
                    f"⚠️ Trans seg {idx} [{trans_start:.2f}-{trans_end:.2f}]: "
                    f"No overlapping diarization segments found!"
                )
            
            # AMÉLIORATION : Pour les premiers segments de transcription, forcer l'alternance si 2 speakers sont détectés
            # mais qu'un speaker apparaît tardivement (après 5 secondes)
            # Cela corrige les cas où pyannote détecte un speaker tardivement alors qu'il parle dès le début
            if idx < 2 and len(first_segment_by_speaker) == 2:
                speaker_ids = sorted(first_segment_by_speaker.keys())
                speaker_0_id, speaker_1_id = speaker_ids[0], speaker_ids[1]
                speaker_0_first = first_segment_by_speaker[speaker_0_id]['start']
                speaker_1_first = first_segment_by_speaker[speaker_1_id]['start']
                
                # Si un speaker apparaît tardivement (après 5 secondes) mais qu'on a forcé 2 speakers,
                # appliquer une alternance forcée pour les 2 premiers segments
                late_speaker = None
                early_speaker = None
                if speaker_0_first > 5.0:
                    late_speaker = speaker_0_id
                    early_speaker = speaker_1_id
                elif speaker_1_first > 5.0:
                    late_speaker = speaker_1_id
                    early_speaker = speaker_0_id
                
                # Si un speaker apparaît tardivement, forcer l'alternance pour les premiers segments
                if late_speaker and early_speaker:
                    # Pour le premier segment (idx=0), forcer l'alternance en ajoutant une intersection
                    # pour le speaker tardif si on a forcé 2 speakers
                    if idx == 0:
                        # Si le segment actuel n'a qu'une intersection avec le speaker qui apparaît tôt,
                        # mais qu'on a forcé 2 speakers, ajouter une intersection pour le speaker tardif
                        # pour permettre l'alternance
                        if (late_speaker not in speaker_intersections and 
                            early_speaker in speaker_intersections and
                            self.num_speakers == 2):  # On a forcé 2 speakers
                            
                            # Ajouter une intersection potentielle pour le speaker tardif
                            # basée sur le fait qu'on sait qu'il y a 2 speakers et qu'ils doivent alterner
                            segment_duration = trans_end - trans_start
                            # Pour le premier segment, donner 60% de la durée au speaker tardif
                            # pour permettre l'alternance (le speaker qui apparaît tôt garde 40%)
                            potential_intersection = segment_duration * 0.6
                            speaker_intersections[late_speaker] = potential_intersection
                            
                            # Réduire légèrement l'intersection du speaker qui apparaît tôt pour équilibrer
                            if early_speaker in speaker_intersections:
                                speaker_intersections[early_speaker] *= 0.7  # Réduire à 70%
                            
                            logger.info(
                                f"🔍 Trans seg {idx}: FORCED ALTERNATION - added intersection for {late_speaker} "
                                f"(late speaker detected at {first_segment_by_speaker[late_speaker]['start']:.2f}s, "
                                f"potential: {potential_intersection:.2f}s) to enable speaker alternation "
                                f"(num_speakers={self.num_speakers} forced)"
                            )
                    elif idx == 1:
                        # Pour le deuxième segment, vérifier si le premier segment a été assigné au speaker tardif
                        # Si oui, alterner avec le speaker qui apparaît tôt
                        if len(segments_with_speakers) > 0:
                            prev_speaker = segments_with_speakers[-1].get('speaker')
                            if prev_speaker == late_speaker:
                                # Le segment précédent était le speaker tardif, forcer l'alternance avec le speaker qui apparaît tôt
                                segment_duration = trans_end - trans_start
                                if early_speaker not in speaker_intersections:
                                    speaker_intersections[early_speaker] = 0.0
                                speaker_intersections[early_speaker] += segment_duration * 0.8
                                logger.info(
                                    f"🔍 Trans seg {idx}: FORCED ALTERNATION - previous was {late_speaker}, "
                                    f"adding intersection for {early_speaker} (potential: {segment_duration * 0.8:.2f}s) to alternate"
                                )
                            elif prev_speaker == early_speaker:
                                # Le segment précédent était le speaker qui apparaît tôt, continuer avec le speaker tardif
                                segment_duration = trans_end - trans_start
                                if late_speaker not in speaker_intersections:
                                    speaker_intersections[late_speaker] = 0.0
                                speaker_intersections[late_speaker] += segment_duration * 0.8
                                logger.info(
                                    f"🔍 Trans seg {idx}: FORCED ALTERNATION - previous was {early_speaker}, "
                                    f"adding intersection for {late_speaker} (potential: {segment_duration * 0.8:.2f}s) to alternate"
                                )
            
            # AMÉLIORATION : Pour les premiers segments de transcription, détecter les changements de speaker
            # basés sur les patterns de conversation (alternance de speakers)
            # Si on a 2 speakers détectés et que le premier segment est assigné au même speaker que le deuxième,
            # mais qu'un autre speaker apparaît peu après, alterner les speakers pour les premiers segments
            elif idx < 3 and len(first_segment_by_speaker) == 2:
                # Si c'est le premier segment (idx=0) et qu'on a déjà assigné un speaker au segment précédent
                # (dans une boucle précédente), vérifier l'alternance
                if idx == 0:
                    # Pour le premier segment, si SPEAKER_01 est détecté dès le début mais SPEAKER_00 apparaît plus tard,
                    # et que SPEAKER_00 apparaît dans les 5 premières secondes, considérer une alternance
                    speaker_ids = sorted(first_segment_by_speaker.keys())
                    if len(speaker_ids) == 2:
                        speaker_0_id, speaker_1_id = speaker_ids[0], speaker_ids[1]
                        speaker_0_first = first_segment_by_speaker[speaker_0_id]['start']
                        speaker_1_first = first_segment_by_speaker[speaker_1_id]['start']
                        
                        # Si les deux speakers apparaissent dans les 5 premières secondes mais avec un décalage
                        # et que le premier segment de transcription commence avant les deux détections
                        if (speaker_0_first < 5.0 and speaker_1_first < 5.0 and 
                            abs(speaker_0_first - speaker_1_first) > 0.5 and
                            trans_start < min(speaker_0_first, speaker_1_first)):
                            
                            # Si SPEAKER_01 a une intersection mais SPEAKER_00 n'en a pas,
                            # et que SPEAKER_00 apparaît très tôt (dans les 2 premières secondes),
                            # considérer une alternance potentielle
                            if (speaker_0_id in speaker_intersections and 
                                speaker_1_id not in speaker_intersections and
                                speaker_1_first < 2.5):
                                
                                # Ajouter une intersection potentielle pour SPEAKER_01 basée sur l'alternance
                                # Plus le segment est proche du début de SPEAKER_01, plus l'intersection est grande
                                time_to_speaker_1 = speaker_1_first - trans_start
                                if time_to_speaker_1 < 3.0:  # SPEAKER_01 apparaît dans les 3 secondes
                                    potential_intersection = max(0.0, (3.0 - time_to_speaker_1) * 0.4)  # Max 1.2s
                                    if speaker_1_id not in speaker_intersections:
                                        speaker_intersections[speaker_1_id] = 0.0
                                    speaker_intersections[speaker_1_id] += potential_intersection
                                    logger.info(
                                        f"🔍 Trans seg {idx}: Added potential intersection for {speaker_1_id} "
                                        f"based on early detection pattern (time to detection: {time_to_speaker_1:.2f}s, "
                                        f"potential: {potential_intersection:.2f}s)"
                                    )
                
                # Pour les segments suivants (idx=1, 2), vérifier l'alternance avec le segment précédent
                elif idx > 0 and len(segments_with_speakers) > 0:
                    prev_speaker = segments_with_speakers[-1].get('speaker')
                    # Si le segment précédent a le même speaker que celui détecté pour ce segment,
                    # mais qu'un autre speaker apparaît peu après, considérer l'alternance
                    if prev_speaker and len(speaker_intersections) > 0:
                        detected_speaker = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                        if prev_speaker == detected_speaker:
                            # Chercher l'autre speaker qui devrait alterner
                            other_speakers = [s for s in unique_diarization_speakers if s != prev_speaker]
                            if other_speakers:
                                other_speaker = other_speakers[0]
                                # Si l'autre speaker apparaît dans les 3 secondes suivantes, considérer l'alternance
                                if other_speaker in first_segment_by_speaker:
                                    other_first = first_segment_by_speaker[other_speaker]['start']
                                    if trans_start < other_first < trans_end + 3.0:
                                        # Ajouter une intersection potentielle pour l'alternance
                                        time_to_other = other_first - trans_start
                                        if time_to_other >= 0:
                                            potential_intersection = max(0.0, (3.0 - time_to_other) * 0.3)
                                            if other_speaker not in speaker_intersections:
                                                speaker_intersections[other_speaker] = 0.0
                                            speaker_intersections[other_speaker] += potential_intersection
                                            logger.info(
                                                f"🔍 Trans seg {idx}: Added potential intersection for {other_speaker} "
                                                f"based on speaker alternation pattern (time to detection: {time_to_other:.2f}s, "
                                                f"potential: {potential_intersection:.2f}s)"
                                            )
            
            # Choisir le speaker avec la plus grande somme d'intersections (comme WhisperX)
            # AMÉLIORATION: Si plusieurs speakers ont des intersections similaires (différence < 30%),
            # privilégier celui qui a le plus de segments distincts (meilleure couverture)
            # AMÉLIORATION: Pour les premiers segments, appliquer une logique d'alternance plus agressive
            if speaker_intersections:
                if len(speaker_intersections) > 1:
                    # Calculer les intersections par speaker
                    sorted_speakers = sorted(speaker_intersections.items(), key=lambda x: x[1], reverse=True)
                    best_speaker, best_intersection = sorted_speakers[0]
                    second_speaker, second_intersection = sorted_speakers[1]
                    
                    # AMÉLIORATION : Pour les 2 premiers segments, appliquer une logique d'alternance
                    # si les deux speakers ont des intersections similaires (ratio > 0.5)
                    intersection_ratio = second_intersection / best_intersection if best_intersection > 0 else 0
                    
                    # Pour les premiers segments, être plus agressif avec l'alternance
                    if idx < 2 and intersection_ratio > 0.5:
                        # Vérifier si on a déjà assigné un speaker au segment précédent
                        if idx > 0 and len(segments_with_speakers) > 0:
                            prev_speaker = segments_with_speakers[-1].get('speaker')
                            # Si le segment précédent a le meilleur speaker, alterner avec le deuxième
                            if prev_speaker == best_speaker:
                                logger.info(
                                    f"🔍 Trans seg {idx}: Applying alternation logic - previous segment was {best_speaker}, "
                                    f"choosing {second_speaker} for alternation (ratio: {intersection_ratio:.2f})"
                                )
                                speaker = second_speaker
                            # Si le segment précédent a le deuxième speaker, alterner avec le meilleur
                            elif prev_speaker == second_speaker:
                                logger.info(
                                    f"🔍 Trans seg {idx}: Applying alternation logic - previous segment was {second_speaker}, "
                                    f"choosing {best_speaker} for alternation (ratio: {intersection_ratio:.2f})"
                                )
                                speaker = best_speaker
                            else:
                                # Pas d'alternance possible, utiliser la logique normale
                                speaker = best_speaker
                        else:
                            # Premier segment : logique spéciale pour forcer l'alternance
                            # Si un speaker apparaît tardivement (après 5s) mais qu'on a forcé 2 speakers,
                            # forcer l'alternance en choisissant le speaker tardif pour le premier segment
                            speaker_ids = sorted(first_segment_by_speaker.keys())
                            if len(speaker_ids) == 2:
                                speaker_0_id, speaker_1_id = speaker_ids[0], speaker_ids[1]
                                speaker_0_first = first_segment_by_speaker[speaker_0_id]['start']
                                speaker_1_first = first_segment_by_speaker[speaker_1_id]['start']
                                
                                # Si un speaker apparaît tardivement (après 5s) et qu'on a forcé 2 speakers,
                                # forcer l'alternance en choisissant le speaker tardif pour le premier segment
                                if (speaker_0_first > 5.0 or speaker_1_first > 5.0) and self.num_speakers == 2:
                                    # Identifier le speaker tardif et le speaker qui apparaît tôt
                                    if speaker_0_first > 5.0:
                                        late_speaker = speaker_0_id
                                        early_speaker = speaker_1_id
                                    else:
                                        late_speaker = speaker_1_id
                                        early_speaker = speaker_0_id
                                    
                                    # Si les intersections sont similaires (ratio > 0.3) OU si le speaker tardif
                                    # a une intersection ajoutée par la logique forcée, choisir le speaker tardif
                                    # pour forcer l'alternance dès le premier segment
                                    if (intersection_ratio > 0.3 or late_speaker in speaker_intersections):
                                        logger.info(
                                            f"🔍 Trans seg {idx}: FIRST SEGMENT - FORCING ALTERNATION - choosing {late_speaker} "
                                            f"(late speaker detected at {first_segment_by_speaker[late_speaker]['start']:.2f}s) "
                                            f"over {early_speaker} (appears at {first_segment_by_speaker[early_speaker]['start']:.2f}s) "
                                            f"to enable speaker alternation (num_speakers={self.num_speakers} forced, ratio: {intersection_ratio:.2f})"
                                        )
                                        speaker = late_speaker
                                    else:
                                        # Si le ratio est trop faible, utiliser la logique normale mais avec un biais
                                        # vers le speaker tardif si on a forcé l'alternance
                                        if late_speaker in speaker_intersections:
                                            speaker = late_speaker
                                            logger.info(
                                                f"🔍 Trans seg {idx}: FIRST SEGMENT - choosing {late_speaker} "
                                                f"(forced alternation intersection present) over {best_speaker}"
                                            )
                                        else:
                                            speaker = best_speaker
                                else:
                                    # Les deux speakers apparaissent tôt, utiliser la logique normale
                                    best_first = first_segment_by_speaker.get(best_speaker, {}).get('start', float('inf'))
                                    second_first = first_segment_by_speaker.get(second_speaker, {}).get('start', float('inf'))
                                    
                                    # Si le deuxième speaker apparaît plus tôt et que les intersections sont similaires,
                                    # le choisir pour le premier segment (corrige les cas où pyannote inverse les speakers)
                                    if second_first < best_first and intersection_ratio > 0.5:
                                        logger.info(
                                            f"🔍 Trans seg {idx}: First segment - choosing {second_speaker} (appears earlier at {second_first:.2f}s) "
                                            f"over {best_speaker} (appears at {best_first:.2f}s) due to similar intersections (ratio: {intersection_ratio:.2f})"
                                        )
                                        speaker = second_speaker
                                    else:
                                        speaker = best_speaker
                            else:
                                speaker = best_speaker
                    elif intersection_ratio > 0.7:  # Si le deuxième speaker a au moins 70% de l'intersection du premier
                        # Compter les segments distincts pour chaque speaker dans cette zone
                        speaker_segment_counts = {}
                        for diar_seg in overlapping_diar_segments:
                            speaker = diar_seg['speaker']
                            if speaker in speaker_intersections:
                                speaker_segment_counts[speaker] = speaker_segment_counts.get(speaker, 0) + 1
                        
                        # Si le deuxième speaker a plus de segments distincts, le choisir
                        if second_speaker in speaker_segment_counts and best_speaker in speaker_segment_counts:
                            if speaker_segment_counts[second_speaker] > speaker_segment_counts[best_speaker]:
                                logger.info(
                                    f"🔍 Trans seg {idx}: Close call! Chose {second_speaker} "
                                    f"(intersection: {second_intersection:.2f}s, segments: {speaker_segment_counts[second_speaker]}) "
                                    f"over {best_speaker} (intersection: {best_intersection:.2f}s, segments: {speaker_segment_counts[best_speaker]}) "
                                    f"due to more distinct segments"
                                )
                                speaker = second_speaker
                            else:
                                speaker = best_speaker
                        else:
                            speaker = best_speaker
                    else:
                        speaker = best_speaker
                    
                    # Logger si on choisit SPEAKER_00 alors qu'il y a plusieurs speakers
                    if speaker == "SPEAKER_00":
                        logger.info(
                            f"🔍 Trans seg {idx}: Chose SPEAKER_00 with {speaker_intersections.get('SPEAKER_00', 0):.2f}s "
                            f"over SPEAKER_01 with {speaker_intersections.get('SPEAKER_01', 0):.2f}s"
                        )
                else:
                    speaker = list(speaker_intersections.keys())[0]
            else:
                # Si aucun overlap, utiliser "UNKNOWN"
                speaker = "UNKNOWN"
            
            # Créer le segment avec le speaker
            seg_with_speaker = trans_seg.copy()
            seg_with_speaker["speaker"] = speaker
            segments_with_speakers.append(seg_with_speaker)
            
            # Compter les assignations par speaker
            speaker_assignment_count[speaker] = speaker_assignment_count.get(speaker, 0) + 1
        
        logger.info(
            f"✅ Assigned speakers to {len(segments_with_speakers)} transcription segments"
        )
        logger.info(f"🔍 Speaker assignment distribution: {speaker_assignment_count}")
        
        # Vérifier si tous les segments ont le même speaker (problème potentiel)
        assigned_speakers = set(seg["speaker"] for seg in segments_with_speakers)
        if len(assigned_speakers) == 1 and len(unique_diarization_speakers) > 1:
            # Calculer la couverture temporelle de la transcription
            if transcription_segments:
                trans_start = min(seg["start"] for seg in transcription_segments)
                trans_end = max(seg["end"] for seg in transcription_segments)
            else:
                trans_start = trans_end = 0.0
            
            # Trouver les segments de diarisation non couverts
            uncovered_speakers = []
            for speaker_id in unique_diarization_speakers:
                if speaker_id not in assigned_speakers:
                    speaker_segments = [s for s in diarization_segments if s["speaker"] == speaker_id]
                    uncovered_duration = sum(s["end"] - s["start"] for s in speaker_segments)
                    uncovered_segments = [s for s in speaker_segments if s["end"] > trans_end or s["start"] < trans_start]
                    if uncovered_segments:
                        uncovered_speakers.append({
                            'speaker': speaker_id,
                            'segments': len(uncovered_segments),
                            'duration': uncovered_duration,
                            'first_segment': uncovered_segments[0],
                            'last_segment': uncovered_segments[-1]
                        })
            
            logger.warning(
                f"⚠️ WARNING: All transcription segments assigned to {assigned_speakers}, "
                f"but diarization detected {len(unique_diarization_speakers)} speakers: {unique_diarization_speakers}"
            )
            if uncovered_speakers:
                logger.warning(
                    f"⚠️ Transcription coverage: [{trans_start:.2f}s - {trans_end:.2f}s], "
                    f"but {len(uncovered_speakers)} speaker(s) have segments outside this range:"
                )
                for uc in uncovered_speakers:
                    logger.warning(
                        f"⚠️   {uc['speaker']}: {uc['segments']} segments ({uc['duration']:.2f}s), "
                        f"first: [{uc['first_segment']['start']:.2f}-{uc['first_segment']['end']:.2f}], "
                        f"last: [{uc['last_segment']['start']:.2f}-{uc['last_segment']['end']:.2f}]"
                    )
                logger.warning(
                    f"⚠️ This suggests VAD/Whisper skipped these audio regions. "
                    f"Consider adjusting VAD parameters (threshold, min_speech_duration_ms) or disabling VAD."
                )
            else:
                logger.warning(f"⚠️ This might indicate a problem with speaker assignment logic")
        
        return segments_with_speakers
    
    @property
    def pipeline(self):
        """
        Propriété pour compatibilité avec l'interface de DiarizationService.
        Retourne le pipeline pyannote.
        """
        return self.model
