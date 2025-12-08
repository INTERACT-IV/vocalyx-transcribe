"""
vocalyx-transcribe/transcribe/audio_utils.py
Utilitaires pour le traitement audio (adapté pour l'architecture microservices)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent

logger = logging.getLogger("vocalyx")

def sanitize_filename(filename: str) -> str:
    """
    Nettoie le nom de fichier pour éviter les injections.
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        str: Nom de fichier sécurisé
    """
    return "".join(c for c in filename if c.isalnum() or c in "._-")

def get_audio_duration(file_path: Path) -> float:
    """
    Obtient la durée réelle de l'audio en secondes.
    Utilise soundfile pour une mesure précise.
    
    Args:
        file_path: Chemin vers le fichier audio
        
    Returns:
        float: Durée en secondes
    """
    try:
        info = sf.info(str(file_path))
        duration = round(info.duration, 2)
        logger.debug(f"Audio duration: {duration}s (via soundfile)")
        return duration
    except Exception as e:
        logger.warning(f"⚠️ Could not get duration with soundfile: {e}")
        # Fallback avec pydub
        try:
            audio = AudioSegment.from_file(str(file_path))
            duration = round(len(audio) / 1000.0, 2)
            logger.debug(f"Audio duration: {duration}s (via pydub)")
            return duration
        except Exception as e2:
            logger.error(f"❌ Could not get duration: {e2}")
            return 0.0

def preprocess_audio(audio_path: Path, preserve_stereo_for_diarization: bool = True) -> Dict[str, Path]:
    """
    Pré-traite l'audio pour améliorer la qualité de transcription.
    - Normalisation du volume
    - Conversion en mono 16kHz pour Whisper
    - Préservation du stéréo pour diarisation (si stéréo détecté)
    
    Args:
        audio_path: Chemin vers le fichier audio original
        preserve_stereo_for_diarization: Si True, préserve une version stéréo pour diarisation
        
    Returns:
        Dict avec les clés:
        - 'mono': Chemin vers la version mono 16kHz (pour Whisper)
        - 'stereo': Chemin vers la version stéréo préservée (None si mono ou si preserve_stereo_for_diarization=False)
        - 'is_stereo': Boolean indiquant si l'audio original était stéréo
    """
    try:
        logger.debug(f"Preprocessing audio: {audio_path.name}")
        audio = AudioSegment.from_file(str(audio_path))
        
        # Détecter si l'audio est stéréo
        is_stereo = audio.channels == 2
        logger.info(f"🔍 Audio format detected: {'STEREO' if is_stereo else 'MONO'} ({audio.channels} channel(s))")
        
        # Normalisation du volume
        audio = normalize(audio)
        
        # Version mono pour Whisper (toujours créée)
        audio_mono = audio.set_channels(1).set_frame_rate(16000)
        mono_path = audio_path.parent / f"{audio_path.stem}_processed_mono.wav"
        audio_mono.export(str(mono_path), format="wav")
        
        result = {
            'mono': mono_path,
            'stereo': None,
            'is_stereo': is_stereo
        }
        
        # Version stéréo pour diarisation (si stéréo détecté et préservation demandée)
        if is_stereo and preserve_stereo_for_diarization:
            # Préserver le stéréo avec normalisation mais sans conversion de sample rate
            audio_stereo = audio.set_frame_rate(16000)  # 16kHz mais stéréo préservé
            stereo_path = audio_path.parent / f"{audio_path.stem}_processed_stereo.wav"
            audio_stereo.export(str(stereo_path), format="wav")
            result['stereo'] = stereo_path
            logger.info(f"✅ Preserved STEREO version for diarization: {stereo_path.name}")
            logger.info(f"   💡 STEREO audio: one channel per speaker (optimized for diarization)")
        else:
            if is_stereo and not preserve_stereo_for_diarization:
                logger.info(f"ℹ️ STEREO detected but preservation disabled")
            else:
                logger.info(f"ℹ️ MONO audio: using mono version for both transcription and diarization")
        
        logger.info(f"✅ Audio preprocessed: {mono_path.name}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ Preprocessing failed, using original: {e}")
        # Fallback : retourner l'original comme mono
        return {
            'mono': audio_path,
            'stereo': None,
            'is_stereo': False
        }

def detect_speech_segments(
    audio_path: Path,
    min_silence_len: int = 500,
    silence_thresh: int = -40
) -> List[Tuple[int, int]]:
    """
    Détecte les segments de parole (Voice Activity Detection).
    
    Args:
        audio_path: Chemin vers le fichier audio
        min_silence_len: Durée minimum de silence en ms (default: 500)
        silence_thresh: Seuil de silence en dB (default: -40)
        
    Returns:
        List[Tuple[int, int]]: Liste de (start_ms, end_ms) des segments avec de la parole
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
        
        speech_segments = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        if not speech_segments:
            # Si aucun segment détecté, retourner l'audio complet
            return [(0, len(audio))]
        
        logger.info(f"🎤 VAD: Detected {len(speech_segments)} speech segments")
        return speech_segments
    except Exception as e:
        logger.warning(f"⚠️ VAD failed, using full audio: {e}")
        audio = AudioSegment.from_file(str(audio_path))
        return [(0, len(audio))]

def split_audio_intelligent(
    file_path: Path,
    use_vad: bool = True,
    segment_length_ms: Optional[int] = None,
    vad_min_silence_len: int = 500,
    vad_silence_thresh: int = -40,
    force_split_for_distribution: bool = False
) -> List[Path]:
    """
    Découpe l'audio de manière intelligente avec taille adaptative.
    Optimisé Phase 3 : Réduction de la consommation mémoire.
    
    Stratégies :
    - Audio court (< 60s) : pas de découpe (sauf si force_split_for_distribution=True)
    - Audio avec VAD : Découpe temporelle simple (faster-whisper gère le VAD intégré)
    - Audio long sans VAD : découpe par durée fixe (adaptative selon CPU)
    
    Args:
        file_path: Chemin vers le fichier audio
        use_vad: Utiliser la détection de voix (default: True)
        segment_length_ms: Longueur des segments en ms (default: None = 45000 si non fourni)
        vad_min_silence_len: VAD - Durée min de silence en ms (non utilisé si use_vad=True, faster-whisper le gère)
        vad_silence_thresh: VAD - Seuil de silence en dB (non utilisé si use_vad=True, faster-whisper le gère)
        force_split_for_distribution: Si True, force la découpe même pour les audios courts (pour distribution)
        
    Returns:
        List[Path]: Liste des chemins vers les segments audio
    """
    segment_paths = []
    
    # Utiliser valeur par défaut si non fournie
    if segment_length_ms is None:
        segment_length_ms = 45000  # Valeur par défaut (sera override par config si disponible)
    
    try:
        # Phase 3 - Optimisation : Obtenir la durée sans charger tout l'audio en mémoire
        try:
            info = sf.info(str(file_path))
            duration_s = info.duration
            duration_ms = int(duration_s * 1000)
        except Exception as e:
            logger.warning(f"⚠️ Could not get duration with soundfile, using pydub: {e}")
            # Fallback : charger avec pydub seulement pour la durée
            audio = AudioSegment.from_file(str(file_path))
            duration_ms = len(audio)
            duration_s = duration_ms / 1000
            del audio  # Libérer immédiatement la mémoire
        
        # Audio court (< 60s) : pas de découpe (sauf si force_split_for_distribution)
        if duration_s < 60 and not force_split_for_distribution:
            logger.info(f"📊 Audio court ({duration_s:.1f}s), pas de découpe")
            return [file_path]
        
        # Si force_split_for_distribution, utiliser une taille de segment plus petite pour les audios courts
        if force_split_for_distribution and duration_s < 60:
            # Pour les audios courts en mode distribué, utiliser des segments de 20-25s
            # Cela permet de mieux distribuer même les audios courts qui génèrent beaucoup de segments Whisper
            adaptive_segment_length_ms = min(segment_length_ms, int(duration_s * 1000 / 2))  # Au moins 2 segments
            adaptive_segment_length_ms = max(adaptive_segment_length_ms, 20000)  # Minimum 20s
            logger.info(f"📊 Audio court ({duration_s:.1f}s) en mode distribué, découpe forcée avec segments de {adaptive_segment_length_ms}ms")
            segment_length_ms = adaptive_segment_length_ms
        
        # Phase 3 - Optimisation : Si VAD activé, utiliser découpe temporelle simple
        # faster-whisper gère déjà le VAD intégré, pas besoin de detect_speech_segments
        # qui charge tout l'audio en mémoire
        if use_vad:
            logger.info(f"🎯 Using time-based segmentation with VAD (faster-whisper will handle VAD filtering)")
            # Découpe temporelle simple : faster-whisper appliquera le VAD sur chaque segment
            # Cela évite de charger tout l'audio deux fois (une fois pour detect_speech_segments, une fois pour export)
            
            # Charger l'audio une seule fois pour la découpe
            audio = AudioSegment.from_file(str(file_path))
            
            # Découper par segments temporels (faster-whisper filtrera le silence)
            for i, start_ms in enumerate(range(0, duration_ms, segment_length_ms)):
                end_ms = min(start_ms + segment_length_ms, duration_ms)
                segment = audio[start_ms:end_ms]
                segment_path = file_path.parent / f"{file_path.stem}_vad{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            
            # Libérer la mémoire immédiatement
            del audio
            
            logger.info(f"🎯 Created {len(segment_paths)} time-based segments (VAD will be applied by faster-whisper)")
            return segment_paths
        
        # Découpe classique par durée (sans VAD)
        logger.info(f"📊 Using time-based segmentation ({segment_length_ms}ms chunks)")
        
        # Charger l'audio une seule fois
        audio = AudioSegment.from_file(str(file_path))
        
        if duration_s < 180:  # Audio moyen (< 3 min) : découper en 2
            mid = duration_ms // 2
            for i, (start, end) in enumerate([(0, mid), (mid, duration_ms)]):
                segment = audio[start:end]
                segment_path = file_path.parent / f"{file_path.stem}_seg{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            logger.info(f"📊 Audio moyen ({duration_s:.1f}s), découpe en 2")
        else:  # Audio long : découper par segments configurables
            for i, start_ms in enumerate(range(0, duration_ms, segment_length_ms)):
                end_ms = min(start_ms + segment_length_ms, duration_ms)
                segment = audio[start_ms:end_ms]
                segment_path = file_path.parent / f"{file_path.stem}_seg{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            logger.info(f"📊 Audio long ({duration_s:.1f}s), découpe en {len(segment_paths)} segments")
        
        # Libérer la mémoire immédiatement
        del audio
        
        return segment_paths
        
    except Exception as e:
        # En cas d'erreur, nettoyer les segments partiels
        for seg_path in segment_paths:
            seg_path.unlink(missing_ok=True)
        logger.error(f"❌ Segmentation error: {e}")
        raise e

def cleanup_segments(segment_paths: List[Path]):
    """
    Nettoie les fichiers de segments temporaires.
    
    Args:
        segment_paths: Liste des chemins vers les segments à supprimer
    """
    for seg_path in segment_paths:
        try:
            if seg_path.exists():
                seg_path.unlink()
                logger.debug(f"🧹 Deleted segment: {seg_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete segment {seg_path.name}: {e}")