# Vocalyx Transcribe

Workers Celery pour la transcription audio avec Whisper et la diarisation des locuteurs.

## Description

Module worker de Vocalyx exécutant les tâches de transcription audio de manière distribuée. Utilise Whisper (OpenAI) pour la transcription et la diarisation stéréo (sans modèle ML) pour l'identification des locuteurs. Implémente un cache de modèles pour optimiser les performances.

## Architecture

### Structure

```
vocalyx-transcribe/
├── worker.py                    # Point d'entrée Celery
├── transcription_service.py     # Service de transcription Whisper
├── audio_utils.py               # Utilitaires de traitement audio
├── stereo_diarization.py        # Service de diarisation stéréo (sans modèle ML)
├── application/
│   └── services/
│       └── transcription_worker_service.py  # Service métier
├── infrastructure/
│   └── api/
│       └── api_client.py        # Client API
└── config.py                    # Configuration
```

### Fonctionnalités

- **Transcription audio** : Conversion audio → texte avec Whisper
- **Diarisation** : Identification et séparation des locuteurs
- **Traitement audio** : VAD (Voice Activity Detection), segmentation
- **Cache de modèles** : Réutilisation des modèles Whisper chargés
- **Traitement parallèle** : Segmentation et transcription en parallèle
- **Monitoring** : Statistiques de performance et santé du worker

## Dépendances principales

### Celery
Système de files d'attente distribuées pour l'exécution asynchrone de tâches. Gère la distribution des transcriptions entre plusieurs workers.

### faster-whisper
Implémentation optimisée de Whisper utilisant CTranslate2. Fournit des performances améliorées par rapport à l'implémentation originale OpenAI.

### soundfile / pydub
Bibliothèques de traitement audio. `soundfile` pour la lecture/écriture de fichiers audio, `pydub` pour les opérations de manipulation audio.

### av (PyAV)
Wrapper Python pour FFmpeg. Utilisé pour le décodage et l'encodage de formats audio/vidéo.

### httpx
Client HTTP pour communiquer avec l'API centrale. Récupère les métadonnées des transcriptions et envoie les résultats.

### psutil
Bibliothèque de monitoring système. Utilisée pour collecter les statistiques CPU/RAM du worker.

### redis
Client Redis pour la connexion au broker Celery. Utilisé pour recevoir et traiter les tâches.

## Configuration

Variables d'environnement principales :

- `INSTANCE_NAME` : Nom d'identification du worker
- `VOCALYX_API_URL` : URL de l'API centrale
- `CELERY_BROKER_URL` : URL du broker Celery
- `CELERY_RESULT_BACKEND` : Backend de résultats Celery
- `WHISPER_MODEL` : Chemin ou nom du modèle Whisper
- `WHISPER_DEVICE` : Device (cpu, cuda)
- `WHISPER_COMPUTE_TYPE` : Type de calcul (int8, float16, float32)
- `WHISPER_LANGUAGE` : Langue par défaut (fr, en, etc.)
- `MAX_WORKERS` : Nombre de workers parallèles
- `VAD_ENABLED` : Activation du VAD
- `LOG_LEVEL` : Niveau de logging

## Tâche Celery

### transcribe_audio_task

Tâche principale exécutée par le worker :

1. **Récupération** : Récupère les métadonnées de la transcription depuis l'API
2. **Préparation** : Charge le modèle Whisper (avec cache)
3. **Traitement audio** : Prétraitement (mono/stéréo, segmentation)
4. **Transcription** : Exécute Whisper sur l'audio
5. **Diarisation** : Optionnellement, identifie les locuteurs
6. **Sauvegarde** : Envoie les résultats à l'API

### Paramètres

- `transcription_id` : Identifiant de la transcription
- `max_retries` : Nombre de tentatives en cas d'échec (3)
- `soft_time_limit` : Limite de temps douce (1800s)
- `time_limit` : Limite de temps dure (2100s)

## Traitement audio

### Préprocessing

- **Détection du format** : Mono ou stéréo
- **Conversion mono** : Pour Whisper (mono requis)
- **Préservation stéréo** : Pour la diarisation (si activée)
- **Normalisation** : Ajustement des niveaux audio

### Segmentation

Segmentation adaptative selon la durée :
- **Court** (< 60s) : Pas de découpe
- **Moyen** (60-300s) : Découpe en 2 segments
- **Long** (> 300s) : Découpe en plusieurs segments

La taille des segments s'adapte également au CPU disponible.

### VAD (Voice Activity Detection)

Détection automatique des segments de parole pour :
- Éviter de transcrire les silences
- Améliorer la précision
- Réduire le temps de traitement

## Cache de modèles

Système de cache LRU pour les modèles Whisper :

- **Limite** : 2 modèles en cache maximum
- **Réutilisation** : Évite le rechargement (5-15s économisées)
- **Gestion mémoire** : Suppression automatique du moins récent

## Diarisation

Service optionnel utilisant la diarisation stéréo (sans modèle ML) pour :

- **Identification des locuteurs** : Séparation par canaux stéréo (gauche = SPEAKER_00, droit = SPEAKER_01)
- **Segmentation temporelle** : Attribution des segments aux locuteurs basée sur la détection de voix par canal
- **Performance** : Très rapide (quelques secondes) car aucun modèle ML requis
- **Requis** : Fichiers audio en stéréo avec un canal par locuteur

## Monitoring

Le worker expose des statistiques via Celery control :

- **CPU** : Pourcentage d'utilisation
- **Mémoire** : RSS et pourcentage
- **Uptime** : Temps de fonctionnement
- **Statistiques métier** : Audio traité, temps de traitement

## Logs

Les logs sont écrits dans `./shared/logs/vocalyx-transcribe-<instance>.log` avec le format :

```
%(asctime)s [%(levelname)s] %(name)s: %(message)s
```

Voir `DOCUMENTATION_LOGS.md` pour la documentation complète des logs.

## Performance

### Optimisations

- **Cache de modèles** : Réduction du temps de chargement
- **Traitement parallèle** : Utilisation de plusieurs workers
- **Segmentation adaptative** : Optimisation selon la durée
- **VAD** : Réduction du temps de traitement

### Ressources

- **Mémoire** : 4-8 GB recommandés par worker
- **CPU** : Multi-core recommandé pour le traitement parallèle
- **Stockage** : Espace pour les modèles (~2-5 GB par modèle)

