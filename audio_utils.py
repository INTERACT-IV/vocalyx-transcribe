"""
vocalyx-transcribe/transcribe/audio_utils.py
Utilitaires pour le traitement audio (adapté pour l'architecture microservices)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

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

def preprocess_audio(audio_path: Path) -> Path:
    """
    Pré-traite l'audio pour améliorer la qualité de transcription.
    - Normalisation du volume
    - Conversion en mono
    - Conversion en 16kHz
    
    Args:
        audio_path: Chemin vers le fichier audio original
        
    Returns:
        Path: Chemin vers le fichier audio traité
    """
    try:
        logger.debug(f"Preprocessing audio: {audio_path.name}")
        audio = AudioSegment.from_file(str(audio_path))
        
        # Normalisation du volume
        audio = normalize(audio)
        
        # Conversion en mono 16kHz (optimal pour Whisper)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export
        output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
        audio.export(str(output_path), format="wav")
        
        logger.info(f"✅ Audio preprocessed: {output_path.name}")
        return output_path
    except Exception as e:
        logger.warning(f"⚠️ Preprocessing failed, using original: {e}")
        return audio_path

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
    segment_length_ms: int = 45000,
    vad_min_silence_len: int = 500,
    vad_silence_thresh: int = -40
) -> List[Path]:
    """
    Découpe l'audio de manière intelligente.
    
    Stratégies :
    - Audio court (< 60s) : pas de découpe
    - Audio moyen avec VAD : découpe selon les segments de parole
    - Audio long sans VAD : découpe par durée fixe
    
    Args:
        file_path: Chemin vers le fichier audio
        use_vad: Utiliser la détection de voix (default: True)
        segment_length_ms: Longueur des segments en ms (default: 45000)
        vad_min_silence_len: VAD - Durée min de silence en ms (default: 500)
        vad_silence_thresh: VAD - Seuil de silence en dB (default: -40)
        
    Returns:
        List[Path]: Liste des chemins vers les segments audio
    """
    segment_paths = []
    
    try:
        audio = AudioSegment.from_file(str(file_path))
        duration_ms = len(audio)
        duration_s = duration_ms / 1000
        
        # Audio court (< 60s) : pas de découpe
        if duration_s < 60:
            logger.info(f"📊 Audio court ({duration_s:.1f}s), pas de découpe")
            return [file_path]
        
        # VAD activé : découper selon les segments de parole
        if use_vad:
            logger.info(f"🎯 Using VAD-based segmentation")
            speech_segments = detect_speech_segments(
                file_path,
                min_silence_len=vad_min_silence_len,
                silence_thresh=vad_silence_thresh
            )
            
            # Grouper les segments proches (< 2s d'écart) pour éviter trop de fragmentation
            merged_segments = []
            current_start, current_end = speech_segments[0]
            
            for start, end in speech_segments[1:]:
                if start - current_end < 2000:  # Si moins de 2s d'écart
                    current_end = end
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_segments.append((current_start, current_end))
            
            # Exporter les segments
            for i, (start_ms, end_ms) in enumerate(merged_segments):
                segment = audio[start_ms:end_ms]
                segment_path = file_path.parent / f"{file_path.stem}_vad{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            
            logger.info(f"🎯 VAD: Created {len(segment_paths)} optimized segments")
            return segment_paths
        
        # Découpe classique par durée (sans VAD)
        logger.info(f"📊 Using time-based segmentation ({segment_length_ms}ms chunks)")
        
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