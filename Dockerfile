FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create entrypoint script with better error handling
RUN echo '#!/bin/bash\n\
set -e\n\
if [ -z "$SCRIPT_NAME" ]; then\n\
    echo "ERROR: SCRIPT_NAME environment variable is required"\n\
    exit 1\n\
fi\n\
echo "Starting DeFi Pipeline - Script: $SCRIPT_NAME"\n\
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"\n\
exec python main_pipeline.py' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && chown appuser:appuser /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["/app/entrypoint.sh"]