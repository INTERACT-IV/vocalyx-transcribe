"""
Module de transcription Vocalyx
Gère la transcription audio via faster-whisper
"""

from .transcription import (
    initialize_whisper_model,
    cleanup_resources,
    run_transcription_optimized,
    whisper_model
)

from .audio_utils import (
    sanitize_filename,
    get_audio_duration,
    preprocess_audio,
    detect_speech_segments,
    split_audio_intelligent
)

__all__ = [
    'initialize_whisper_model',
    'cleanup_resources',
    'run_transcription_optimized',
    'whisper_model',
    'sanitize_filename',
    'get_audio_duration',
    'preprocess_audio',
    'detect_speech_segments',
    'split_audio_intelligent'
]