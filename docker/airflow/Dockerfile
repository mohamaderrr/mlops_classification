FROM apache/airflow:2.9.1-python3.10


USER root

# Installer Docker CLI
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Installer les dépendances Python
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r /requirements.txt

# Copier le code source
COPY src/ /opt/airflow/src/
COPY configs/ /opt/airflow/configs/

# Installer DVC
RUN pip install dvc[s3] boto3
RUN pip install mlflow==2.8.1

# Configuration DVC
ENV DVC_CACHE_TYPE=symlink