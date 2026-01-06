"""
Service pour fusionner les résultats de diarisation distribuée
"""

import logging
from typing import List, Dict

logger = logging.getLogger("vocalyx")


class DiarizationMerger:
    """Fusionne les résultats de diarisation distribuée en gérant la continuité des locuteurs"""
    
    @staticmethod
    def merge_diarization_segments(
        diarization_results: List[Dict],
        time_offsets: List[float]
    ) -> List[Dict]:
        """
        Fusionne les segments de diarisation de plusieurs workers en gérant la continuité des locuteurs.
        
        Args:
            diarization_results: Liste de résultats de diarisation par segment
                Chaque élément est une liste de segments avec start/end/speaker
            time_offsets: Liste des offsets temporels pour chaque segment (début de chaque segment)
            
        Returns:
            Liste fusionnée de segments de diarisation avec timestamps ajustés
        """
        if not diarization_results:
            return []
        
        merged_segments = []
        
        # Dictionnaire pour mapper les locuteurs entre segments
        # Format: {segment_index: {old_speaker: new_speaker}}
        speaker_mapping = {}
        
        # Compteur global pour les nouveaux identifiants de locuteurs
        next_speaker_id = 0
        speaker_name_map = {}  # {original_speaker: global_speaker}
        
        for segment_idx, (segments, time_offset) in enumerate(zip(diarization_results, time_offsets)):
            if not segments:
                continue
            
            # Pour le premier segment, on garde les locuteurs tels quels
            if segment_idx == 0:
                # Créer le mapping initial
                unique_speakers = set(seg["speaker"] for seg in segments)
                for speaker in sorted(unique_speakers):
                    global_speaker = f"SPEAKER_{next_speaker_id:02d}"
                    speaker_name_map[speaker] = global_speaker
                    next_speaker_id += 1
            else:
                # Pour les segments suivants, on doit mapper les locuteurs
                # en fonction de la continuité avec le segment précédent
                prev_segments = diarization_results[segment_idx - 1]
                if prev_segments and len(prev_segments) > 0:
                    # Trouver le dernier locuteur du segment précédent
                    last_speaker = prev_segments[-1]["speaker"]
                    last_speaker_global = speaker_name_map.get(last_speaker, None)
                    
                    # Trouver le premier locuteur du segment actuel
                    if segments and len(segments) > 0:
                        first_speaker = segments[0]["speaker"]
                        
                        # Si le premier locuteur du segment actuel correspond au dernier du précédent
                        # (en tenant compte d'un petit chevauchement), on les mappe ensemble
                        if last_speaker_global:
                            # Vérifier s'il y a continuité temporelle
                            prev_end = prev_segments[-1]["end"] + time_offsets[segment_idx - 1]
                            curr_start = segments[0]["start"] + time_offset
                            
                            # Si le début du segment actuel est proche de la fin du précédent (< 2s)
                            if abs(curr_start - prev_end) < 2.0:
                                # Mapper le premier locuteur du segment actuel au dernier du précédent
                                if first_speaker not in speaker_name_map:
                                    speaker_name_map[first_speaker] = last_speaker_global
                    
                    # Mapper les autres locuteurs
                    unique_speakers = set(seg["speaker"] for seg in segments)
                    for speaker in sorted(unique_speakers):
                        if speaker not in speaker_name_map:
                            global_speaker = f"SPEAKER_{next_speaker_id:02d}"
                            speaker_name_map[speaker] = global_speaker
                            next_speaker_id += 1
            
            # Ajouter les segments avec timestamps ajustés et locuteurs mappés
            for seg in segments:
                merged_segments.append({
                    "start": round(seg["start"] + time_offset, 2),
                    "end": round(seg["end"] + time_offset, 2),
                    "speaker": speaker_name_map.get(seg["speaker"], seg["speaker"])
                })
        
        # Trier par timestamp
        merged_segments.sort(key=lambda x: x["start"])
        
        logger.info(
            f"✅ Merged {len(merged_segments)} diarization segments from {len(diarization_results)} workers"
        )
        
        return merged_segments

