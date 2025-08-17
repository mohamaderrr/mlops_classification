FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système pour compiler les extensions Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        python3-dev \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/api/ ./
COPY src/models/ ./models_src/

# Créer les répertoires nécessaires
RUN mkdir -p models data/processed

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
