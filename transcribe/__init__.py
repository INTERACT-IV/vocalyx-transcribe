"""
Module de transcription Vocalyx
Gère la transcription audio via faster-whisper
"""

from .transcription import (
    TranscriptionService
)

from .audio_utils import (
    sanitize_filename,
    get_audio_duration,
    preprocess_audio,
    detect_speech_segments,
    split_audio_intelligent
)

__all__ = [
    'TranscriptionService',
    'sanitize_filename',
    'get_audio_duration',
    'preprocess_audio',
    'detect_speech_segments',
    'split_audio_intelligent'
]