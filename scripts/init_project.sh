#!/bin/bash

# Script d'initialisation du projet MLOps

set -e

echo "🚀 Initialisation du projet MLOps..."

# Créer la structure des répertoires
echo "📁 Création de la structure des répertoires..."
mkdir -p data/{raw,processed,monitoring}
mkdir -p models
mkdir -p reports
mkdir -p logs
mkdir -p .dvc

# Initialiser Git si pas déjà fait
if [ ! -d ".git" ]; then
    echo "🔧 Initialisation de Git..."
    git init
    git add .gitignore
    git commit -m "Initial commit"
fi

# Initialiser DVC
echo "🔧 Initialisation de DVC..."
dvc init

# Configurer DVC remote (MinIO)
echo "🔧 Configuration du remote DVC..."
dvc remote add -d minio s3://mlops-bucket
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# Créer les environnements virtuels
echo "🐍 Configuration de l'environnement Python..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Générer la clé Fernet pour Airflow
echo "🔑 Génération de la clé Fernet pour Airflow..."
python -c "from cryptography.fernet import Fernet; print('AIRFLOW_FERNET_KEY=' + Fernet.generate_key().decode())" > .env

# Initialiser la base de données Airflow
echo "🗄️  Initialisation d'Airflow..."
export AIRFLOW_HOME=$PWD/airflow
mkdir -p $AIRFLOW_HOME
airflow db init

# Créer un utilisateur admin Airflow
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "✅ Initialisation terminée!"
echo "📋 Prochaines étapes:"
echo "   1. Configurer les secrets GitHub pour CI/CD"
echo "   2. Adapter les URLs dans les fichiers de config"
echo "   3. Lancer: docker-compose up -d"
echo "   4. Accéder à Airflow: http://localhost:8080"
echo "   5. Accéder à Grafana: http://localhost:3000"