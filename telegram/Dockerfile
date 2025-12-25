FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from telegram directory
COPY telegram/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir matplotlib pandas

# Copy shared files from root (needed by telegram_reporter.py)
COPY config_loader.py /app/

# Copy db_utils.py from bot directory (since it's there in your structure)
COPY bot/db_utils.py /app/

# Copy telegram reporter
COPY telegram/telegram_reporter.py /app/

# Create logs directory
RUN mkdir -p /app/logs

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the telegram reporter
CMD ["python", "telegram_reporter.py"]