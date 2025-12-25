FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared configuration files
COPY config_loader.py /app/
COPY strategy_config.ini /app/

# Copy instruments.json
COPY bot/instruments.json /app/instruments.json
COPY bot/instruments.json /app/bot/instruments.json

# Copy telegram reporter
COPY telegram /app/app_telegram

# Copy bot
COPY bot /app/bot

RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

CMD ["python", "bot/bot_main.py"]
