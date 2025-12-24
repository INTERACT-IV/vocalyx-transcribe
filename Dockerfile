# vocalyx-transcribe/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système pour audio
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .
# Créer les répertoires nécessaires
RUN mkdir -p /app/logs /app/models /app/shared_uploads

# Télécharger le modèle Whisper (optionnel, peut être monté en volume)
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='/app/models')"

# Commande de démarrage du worker Celery
CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]