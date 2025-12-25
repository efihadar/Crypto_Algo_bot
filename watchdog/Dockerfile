FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    psycopg2-binary \
    requests \
    loguru

# Copy watchdog script
COPY watchdog/watchdog.py /app/watchdog.py

# Copy db_utils because watchdog uses database
COPY bot/db_utils.py /app/db_utils.py

# Copy config loader
COPY config_loader.py /app/config_loader.py

CMD ["python", "/app/watchdog.py"]
