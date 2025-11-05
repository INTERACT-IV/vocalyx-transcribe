# vocalyx-transcribe

Worker Celery pour la transcription audio avec Faster-Whisper.

## 🎯 Rôle

- **Consommateur** des tâches Celery depuis la queue Redis
- Exécution des transcriptions avec Faster-Whisper
- Communication avec `vocalyx-api` via HTTP pour récupérer et mettre à jour les transcriptions
- Scalable horizontalement (plusieurs workers possibles)

## 🏗️ Architecture

```
vocalyx-transcribe/
├── transcribe/
│   ├── __init__.py
│   └── audio_utils.py         # Utilitaires audio (VAD, découpe)
├── logs/                       # Répertoire des logs
├── models/                     # Modèles Whisper (peut être monté en volume)
├── shared_uploads/             # Uploads partagés avec l'API
├── worker.py                   # Point d'entrée Celery Worker
├── api_client.py               # Client HTTP vers vocalyx-api
├── transcription_service.py    # Service de transcription Whisper
├── config.py                   # Configuration
├── logging_config.py           # Configuration du logging
├── requirements.txt
├── Dockerfile
└── config.ini
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- Redis (pour Celery)
- vocalyx-api en cours d'exécution
- FFmpeg (pour le traitement audio)

### Installation locale

```bash
# Cloner le dépôt
git clone <repository>
cd vocalyx-transcribe

# Créer un environnement virtuel
python3.10 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle Whisper (si pas déjà fait)
# Il sera téléchargé automatiquement au premier lancement

# Configurer
cp config.ini config.local.ini
# Éditer config.local.ini

# Lancer le worker
python worker.py

# OU avec Celery directement
celery -A worker.celery_app worker --loglevel=info --concurrency=2
```

## 🐳 Docker

```bash
# Build
docker build -t vocalyx-transcribe .

# Run
docker run \
  -e CELERY_BROKER_URL="redis://redis:6379/0" \
  -e VOCALYX_API_URL="http://vocalyx-api:8000" \
  -v $(pwd)/shared_uploads:/app/shared_uploads \
  -v $(pwd)/models:/app/models \
  vocalyx-transcribe
```

## ⚙️ Configuration

### Paramètres Principaux

#### Model Whisper
```ini
[WHISPER]
model = ./models/openai-whisper-small  # tiny, base, small, medium, large-v3
device = cpu                            # cpu, cuda
compute_type = int8                     # int8, float16, float32
language = fr                           # fr, en, es, etc.
```

#### Performance
```ini
[PERFORMANCE]
max_workers = 2          # Concurrence Celery
vad_enabled = true       # Voice Activity Detection
beam_size = 5            # Qualité du décodage
```

#### API
```ini
[API]
url = http://localhost:8000    # URL de vocalyx-api
```

#### Celery
```ini
[CELERY]
broker_url = redis://localhost:6379/0
result_backend = redis://localhost:6379/0
```

## 🔄 Flux de Travail

```
1. Worker démarre et se connecte à Redis
2. Worker attend une tâche "transcribe_audio"
3. Tâche reçue avec transcription_id
4. Worker → API: GET /api/transcriptions/{id} (récupérer infos)
5. Worker → API: PATCH /api/transcriptions/{id} (status=processing)
6. Worker exécute la transcription Whisper
7. Worker → API: PATCH /api/transcriptions/{id} (résultats + status=done)
8. Worker attend la prochaine tâche
```

## 📊 Monitoring

### Logs
```bash
# Logs du worker
tail -f logs/vocalyx-transcribe.log
```

### Celery Flower (optionnel)
```bash
# Démarrer Flower pour monitoring web
celery -A worker.celery_app flower --port=5555

# Accéder: http://localhost:5555
```

### Commandes Celery Utiles
```bash
# Voir les workers actifs
celery -A worker.celery_app inspect active

# Voir les tâches en attente
celery -A worker.celery_app inspect reserved

# Statistiques
celery -A worker.celery_app inspect stats

# Arrêter tous les workers
celery -A worker.celery_app control shutdown
```

## 🔧 Scalabilité

### Lancer plusieurs workers

```bash
# Worker 1
INSTANCE_NAME=worker-01 python worker.py

# Worker 2 (autre terminal)
INSTANCE_NAME=worker-02 python worker.py

# Worker 3 (autre terminal)
INSTANCE_NAME=worker-03 python worker.py
```

Ou avec Docker Compose (voir docker-compose.yml dans la racine).

### Stratégies de Scalabilité

1. **Horizontal** : Ajouter plus de workers
2. **Vertical** : Augmenter `max_workers` (concurrence Celery)
3. **GPU** : Utiliser `device=cuda` pour des transcriptions plus rapides

## 🚨 Gestion des Erreurs

Le worker gère automatiquement :
- **Retry** : 3 tentatives avec 60s entre chaque
- **Crash** : Celery re-enqueue automatiquement la tâche
- **API indisponible** : Le worker continue mais logge l'erreur
- **Fichier manquant** : Marque la transcription en erreur

## 🔒 Sécurité

### Communication avec l'API

Le worker utilise une clé interne (`X-Internal-Key`) pour communiquer avec vocalyx-api.

```ini
[SECURITY]
internal_api_key = SECRET_KEY_HERE
```

**⚠️ Cette clé DOIT être identique à celle configurée dans vocalyx-api.**

## 📝 Changelog

### Version 1.0.0
- Architecture microservices (worker Celery)
- Communication HTTP avec vocalyx-api
- Plus d'accès direct à la base de données
- Support multi-workers natif

## 📄 Licence

Propriétaire - Guilhem RICHARD