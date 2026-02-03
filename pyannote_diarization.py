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
    
    Args:
        audio_path: Chemin vers le fichier audio
        sample_rate: Sample rate cible (défaut: 16000)
        
    Returns:
        numpy.ndarray: Audio en mono, normalisé entre -1 et 1
    """
    try:
        # Charger avec soundfile (compatible avec WhisperX)
        audio, sr = sf.read(str(audio_path))
        
        # Convertir en mono si stéréo (pyannote accepte mono et stéréo, mais on normalise en mono)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample si nécessaire
        if sr != sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)
        
        # Normaliser entre -1 et 1
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
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
        
        try:
            # Essayer d'abord avec 'token' (versions récentes), puis 'use_auth_token' (anciennes versions)
            # Si aucun token n'est fourni, pyannote essaiera automatiquement les variables d'environnement
            if auth_token:
                logger.info("🔑 Using HuggingFace token for authentication")
                # Essayer d'abord 'token' (versions récentes de pyannote.audio)
                try:
                    self.model = Pipeline.from_pretrained(
                        model_name,
                        token=auth_token
                    ).to(device)
                except TypeError as e:
                    # Si 'token' n'est pas supporté, essayer 'use_auth_token' (anciennes versions)
                    if "unexpected keyword argument 'token'" in str(e):
                        logger.info("🔄 Trying with 'use_auth_token' parameter (older pyannote.audio version)")
                        self.model = Pipeline.from_pretrained(
                            model_name,
                            use_auth_token=auth_token
                        ).to(device)
                    else:
                        raise
            else:
                logger.info("ℹ️ No token provided, pyannote will use environment variables if available")
                # Essayer sans token (pyannote utilisera les variables d'environnement)
                self.model = Pipeline.from_pretrained(
                    model_name
                ).to(device)
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
            diarization_result = self.model(
                audio_data,
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            
            # Gérer différents formats de retour selon la version de pyannote.audio
            # Les versions récentes peuvent retourner DiarizeOutput au lieu d'Annotation directement
            diarization = None
            
            # Essayer différentes méthodes pour accéder à l'Annotation
            # 1. Vérifier si c'est directement une Annotation (format classique)
            if hasattr(diarization_result, 'itertracks'):
                diarization = diarization_result
                logger.debug("✅ Using direct Annotation format")
            # 2. Essayer d'accéder directement à l'attribut 'annotation' (DiarizeOutput)
            # Même si hasattr retourne False, l'attribut peut exister via __getattr__
            else:
                try:
                    # Essayer directement l'accès à l'attribut
                    diarization = diarization_result.annotation
                    if hasattr(diarization, 'itertracks'):
                        logger.debug("✅ Using DiarizeOutput.annotation (direct access)")
                except AttributeError:
                    # Si l'attribut n'existe pas, essayer getattr
                    try:
                        diarization = getattr(diarization_result, 'annotation', None)
                        if diarization and hasattr(diarization, 'itertracks'):
                            logger.debug("✅ Found annotation via getattr")
                    except:
                        pass
            
            # Si toujours None, logger pour debug et essayer toutes les méthodes possibles
            if diarization is None:
                logger.warning(f"⚠️ Could not find Annotation in result type: {type(diarization_result)}")
                all_attrs = [a for a in dir(diarization_result) if not a.startswith('_')]
                logger.info(f"🔍 Available attributes: {all_attrs}")
                
                # Essayer d'accéder aux attributs qui contiennent l'annotation
                # Priorité: speaker_diarization (standard avec chevauchements) > exclusive_speaker_diarization (sans chevauchements)
                priority_attrs = ['speaker_diarization', 'exclusive_speaker_diarization', 'annotation', 'diarization']
                
                for attr_name in priority_attrs:
                    if attr_name in all_attrs:
                        try:
                            attr_value = getattr(diarization_result, attr_name)
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
        
        # Créer une liste des segments avec speakers assignés
        segments_with_speakers = []
        speaker_assignment_count = {}
        
        # Utiliser la même logique que WhisperX : sommer les intersections par speaker
        # C'est plus simple et plus efficace que de chercher le meilleur segment individuel
        for idx, trans_seg in enumerate(transcription_segments):
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            
            # Calculer l'intersection avec chaque segment de diarisation et grouper par speaker
            speaker_intersections = {}
            
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
            
            # Logger pour debug sur quelques segments problématiques (où SPEAKER_01 devrait apparaître)
            if len(speaker_intersections) > 1 and idx in [14, 15, 16, 25, 26, 27]:
                logger.info(
                    f"🔍 Trans seg {idx} [{trans_start:.2f}-{trans_end:.2f}]: "
                    f"intersections by speaker: {speaker_intersections}"
                )
            
            # Logger pour debug sur quelques segments problématiques (où SPEAKER_01 devrait apparaître)
            if len(speaker_intersections) > 1 and idx in [14, 15, 16, 25, 26, 27]:
                logger.info(
                    f"🔍 Trans seg {idx} [{trans_start:.2f}-{trans_end:.2f}]: "
                    f"intersections by speaker: {speaker_intersections}"
                )
            
            # Choisir le speaker avec la plus grande somme d'intersections (comme WhisperX)
            if speaker_intersections:
                speaker = max(speaker_intersections.items(), key=lambda x: x[1])[0]
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
            logger.warning(
                f"⚠️ WARNING: All transcription segments assigned to {assigned_speakers}, "
                f"but diarization detected {len(unique_diarization_speakers)} speakers: {unique_diarization_speakers}"
            )
            logger.warning(f"⚠️ This might indicate a problem with speaker assignment logic")
        
        return segments_with_speakers
    
    @property
    def pipeline(self):
        """
        Propriété pour compatibilité avec l'interface de DiarizationService.
        Retourne le pipeline pyannote.
        """
        return self.model
