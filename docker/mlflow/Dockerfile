# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_HOME=/mlflow

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR ${MLFLOW_HOME}

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose MLflow default port
EXPOSE 5000

# Set default command to run MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "s3://mlflow-artifacts"]
