"""
vocalyx-transcribe/stereo_diarization.py
Service de diarisation basé sur la séparation stéréo (légère et rapide)

Cette solution est optimale lorsque tous les fichiers audio sont en stéréo
avec un canal par speaker (canal gauche = speaker 0, canal droit = speaker 1).

AVANTAGES vs pyannote.audio:
- 100-1000x plus rapide (quelques secondes vs plusieurs minutes)
- Aucun modèle ML requis (pas de CPU lourd)
- Utilise uniquement la séparation de canaux et VAD simple
- Parfait pour le cas d'usage stéréo dédié
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent

logger = logging.getLogger("vocalyx")


class StereoDiarizationService:
    """
    Service de diarisation basé sur la séparation stéréo.
    
    Stratégie:
    1. Extraire les canaux gauche et droit
    2. Détecter la présence de voix sur chaque canal avec VAD
    3. Assigner speaker 0 au canal gauche, speaker 1 au canal droit
    4. Créer les segments de diarisation basés sur les timestamps
    """
    
    def __init__(self, config=None):
        """
        Initialise le service de diarisation stéréo.
        
        Args:
            config: Configuration (optionnel, pour compatibilité avec l'interface existante)
        """
        self.config = config
        logger.info("✅ Stereo diarization service initialized (lightweight, no ML models required)")
    
    def diarize(self, audio_path: Path) -> List[Dict[str, float]]:
        """
        Effectue la diarisation stéréo sur un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio (doit être stéréo)
            
        Returns:
            Liste de dictionnaires avec les segments de chaque locuteur:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        try:
            # Vérifier que le fichier est stéréo
            audio = AudioSegment.from_file(str(audio_path))
            if audio.channels != 2:
                logger.warning(
                    f"⚠️ Audio is not stereo ({audio.channels} channel(s)). "
                    "Stereo diarization requires 2 channels. Returning empty result."
                )
                return []
            
            logger.info(f"🎤 Running stereo diarization on {audio_path.name} (duration: {len(audio)/1000:.1f}s)...")
            
            # Extraire les canaux gauche et droit
            left_channel = audio.split_to_mono()[0]
            right_channel = audio.split_to_mono()[1]
            
            # Normaliser les canaux pour une meilleure détection
            left_channel = normalize(left_channel)
            right_channel = normalize(right_channel)
            
            # Détecter les segments de parole sur chaque canal
            # Paramètres VAD: seuil de silence et durée minimale
            silence_thresh = getattr(self.config, 'stereo_diarization_silence_thresh', -40)
            min_speech_duration_ms = getattr(self.config, 'stereo_diarization_min_speech_ms', 250)
            
            left_speech_segments = detect_nonsilent(
                left_channel,
                min_silence_len=min_speech_duration_ms,
                silence_thresh=silence_thresh
            )
            right_speech_segments = detect_nonsilent(
                right_channel,
                min_silence_len=min_speech_duration_ms,
                silence_thresh=silence_thresh
            )
            
            logger.info(
                f"🔍 Detected {len(left_speech_segments)} speech segments on left channel (SPEAKER_00), "
                f"{len(right_speech_segments)} on right channel (SPEAKER_01)"
            )
            
            # Convertir les segments en secondes et créer les segments de diarisation
            diarization_segments = []
            
            # Traiter le canal gauche (SPEAKER_00)
            for start_ms, end_ms in left_speech_segments:
                diarization_segments.append({
                    "start": round(start_ms / 1000.0, 2),
                    "end": round(end_ms / 1000.0, 2),
                    "speaker": "SPEAKER_00"
                })
            
            # Traiter le canal droit (SPEAKER_01)
            for start_ms, end_ms in right_speech_segments:
                diarization_segments.append({
                    "start": round(start_ms / 1000.0, 2),
                    "end": round(end_ms / 1000.0, 2),
                    "speaker": "SPEAKER_01"
                })
            
            # Trier par timestamp de début
            diarization_segments.sort(key=lambda x: x['start'])
            
            # Fusionner les segments adjacents du même speaker pour éviter la fragmentation
            diarization_segments = self._merge_adjacent_segments(diarization_segments)
            
            # Compter le nombre de locuteurs uniques
            unique_speakers = set(seg["speaker"] for seg in diarization_segments)
            logger.info(
                f"✅ Stereo diarization completed: {len(diarization_segments)} segments, "
                f"{len(unique_speakers)} speaker(s) detected"
            )
            
            return diarization_segments
            
        except Exception as e:
            logger.error(f"❌ Error during stereo diarization: {e}", exc_info=True)
            return []
    
    def _merge_adjacent_segments(
        self, 
        segments: List[Dict[str, float]], 
        max_gap_ms: float = 500.0
    ) -> List[Dict[str, float]]:
        """
        Fusionne les segments adjacents du même speaker si la pause est courte.
        
        Args:
            segments: Liste de segments de diarisation
            max_gap_ms: Pause maximale en ms pour fusionner (défaut: 500ms)
            
        Returns:
            Liste de segments fusionnés
        """
        if not segments:
            return segments
        
        # Grouper par speaker
        segments_by_speaker = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(seg)
        
        merged_segments = []
        
        # Fusionner les segments de chaque speaker
        for speaker, speaker_segments in segments_by_speaker.items():
            # Trier par timestamp
            speaker_segments.sort(key=lambda x: x['start'])
            
            if not speaker_segments:
                continue
            
            # Commencer avec le premier segment
            current_seg = speaker_segments[0].copy()
            
            for next_seg in speaker_segments[1:]:
                gap_ms = (next_seg['start'] - current_seg['end']) * 1000
                
                # Si la pause est courte, fusionner
                if gap_ms <= max_gap_ms:
                    current_seg['end'] = next_seg['end']
                else:
                    # Sauvegarder le segment actuel et passer au suivant
                    merged_segments.append(current_seg)
                    current_seg = next_seg.copy()
            
            # Ajouter le dernier segment
            merged_segments.append(current_seg)
        
        # Trier tous les segments fusionnés par timestamp
        merged_segments.sort(key=lambda x: x['start'])
        
        return merged_segments
    
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
    
    @property
    def pipeline(self):
        """
        Propriété pour compatibilité avec l'interface de DiarizationService.
        Retourne toujours True car ce service n'utilise pas de pipeline ML.
        """
        return True

