"""
Utilitaires pour le traitement audio
"""

import logging
import soundfile as sf
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent

from config import Config

config = Config()
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Nettoie le nom de fichier"""
    return "".join(c for c in filename if c.isalnum() or c in "._-")

def get_audio_duration(file_path: Path) -> float:
    """
    Obtient la durée réelle de l'audio en secondes.
    FIXE: Utilise soundfile pour une mesure précise.
    """
    try:
        info = sf.info(str(file_path))
        return round(info.duration, 2)
    except Exception as e:
        logger.warning(f"⚠️ Could not get duration with soundfile: {e}")
        # Fallback avec pydub
        try:
            audio = AudioSegment.from_file(str(file_path))
            return round(len(audio) / 1000.0, 2)
        except Exception as e2:
            logger.error(f"❌ Could not get duration: {e2}")
            return 0.0

def preprocess_audio(audio_path: Path) -> Path:
    """
    Pré-traite l'audio pour améliorer la qualité de transcription.
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
        
        # Normalisation du volume
        audio = normalize(audio)
        
        # Conversion en mono 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export
        output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
        audio.export(str(output_path), format="wav")
        
        logger.info(f"✅ Audio preprocessed: {output_path.name}")
        return output_path
    except Exception as e:
        logger.warning(f"⚠️ Preprocessing failed, using original: {e}")
        return audio_path

def detect_speech_segments(audio_path: Path) -> List[Tuple[int, int]]:
    """
    Détecte les segments de parole (VAD).
    Retourne une liste de (start_ms, end_ms) des segments avec de la parole.
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
        
        speech_segments = detect_nonsilent(
            audio,
            min_silence_len=config.vad_min_silence_len,
            silence_thresh=config.vad_silence_thresh
        )
        
        if not speech_segments:
            return [(0, len(audio))]
        
        logger.info(f"🎤 VAD: Detected {len(speech_segments)} speech segments")
        return speech_segments
    except Exception as e:
        logger.warning(f"⚠️ VAD failed, using full audio: {e}")
        audio = AudioSegment.from_file(str(audio_path))
        return [(0, len(audio))]

def split_audio_intelligent(file_path: Path, use_vad: bool = True) -> List[Path]:
    """
    Découpe l'audio de manière intelligente.
    """
    segment_paths = []
    audio = AudioSegment.from_file(str(file_path))
    duration_ms = len(audio)
    duration_s = duration_ms / 1000
    
    try:
        # Audio court: pas de découpe
        if duration_s < 60:
            logger.info(f"📊 Audio court ({duration_s:.1f}s), pas de découpe")
            return [file_path]
        
        # VAD activé: découper selon les segments de parole
        if use_vad and config.vad_enabled:
            speech_segments = detect_speech_segments(file_path)
            
            # Grouper les segments proches (< 2s d'écart)
            merged_segments = []
            current_start, current_end = speech_segments[0]
            
            for start, end in speech_segments[1:]:
                if start - current_end < 2000:
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
        
        # Découpe classique par durée
        if duration_s < 180:
            # Audio moyen: découper en 2
            mid = duration_ms // 2
            for i, (start, end) in enumerate([(0, mid), (mid, duration_ms)]):
                segment = audio[start:end]
                segment_path = file_path.parent / f"{file_path.stem}_seg{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            logger.info(f"📊 Audio moyen ({duration_s:.1f}s), découpe en 2")
        else:
            # Audio long: découper par segments configurables
            for i, start_ms in enumerate(range(0, duration_ms, config.segment_length_ms)):
                segment = audio[start_ms:start_ms + config.segment_length_ms]
                segment_path = file_path.parent / f"{file_path.stem}_seg{i}.wav"
                segment.export(str(segment_path), format="wav")
                segment_paths.append(segment_path)
            logger.info(f"📊 Audio long ({duration_s:.1f}s), découpe en {len(segment_paths)}")
        
        return segment_paths
        
    except Exception as e:
        for seg_path in segment_paths:
            seg_path.unlink(missing_ok=True)
        raise e