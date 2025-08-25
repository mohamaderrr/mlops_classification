# MLflow Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies directly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary==2.9.7 \
    boto3==1.29.7 \
    pymysql==1.1.0 \
    cryptography==41.0.7 \
    protobuf==4.24.4 \
    click>=8.0

# Verify mlflow installation
RUN which mlflow && mlflow --version

# Create mlflow user and directories
RUN useradd -m -u 1001 mlflow && \
    mkdir -p /app/mlflow && \
    chown -R mlflow:mlflow /app

# Switch to mlflow user
USER mlflow

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]