FROM python:3.11-slim

# Install cron
RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Paris

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY *.py ./
COPY config.yaml prompts.yaml ./

# Create directories
RUN mkdir -p /app/results /app/logs

# Add crontab — runs daily at 09:00 UTC, logs to /app/logs/cron.log
RUN echo "46 10 * * * cd /app && /usr/local/bin/python3 pipeline.py run >> /app/logs/cron.log 2>&1" | crontab -

CMD ["sh", "-c", "printenv > /etc/environment && cron -f"]
