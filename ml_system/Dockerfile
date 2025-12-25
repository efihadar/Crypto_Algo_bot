# Multi-stage Dockerfile for ML services
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ /app/shared/
COPY ml/ /app/ml/

# ================================
# API Service
# ================================
FROM base as api
CMD ["python", "-m", "ml_system.ml_service"]

# ================================
# Trainer Service
# ================================
FROM base as trainer
CMD ["python", "-m", "ml_system.training.trainer"]

# ================================
# Collector Service
# ================================
FROM base as collector
CMD ["python", "-m", "ml_system.data_collector"]

# ================================
# Monitor Service
# ================================
FROM base as monitor
CMD ["python", "-m", "ml.monitoring.performance_monitor"]