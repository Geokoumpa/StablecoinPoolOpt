FROM python:3.11.11-slim

WORKDIR /app

# Install system dependencies including Playwright browser dependencies
# Combined into a single RUN command to reduce image layers
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    # Playwright browser dependencies
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set up Playwright environment
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN mkdir -p /ms-playwright && chown appuser:appuser /ms-playwright

# Install Playwright dependencies (needs root)
RUN python -m pip install playwright && \
    python -m playwright install-deps chromium

# Copy requirements and install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (as root, but owned by appuser for access)
RUN python -m playwright install chromium \
    && chown -R appuser:appuser /ms-playwright

# Copy application code
COPY --chown=appuser:appuser . .

# Set Python path to include the application directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create entrypoint script with universal pipeline runner
RUN echo '#!/bin/bash\n\
set -e\n\
if [ -z "$SCRIPT_NAME" ]; then\n\
    echo "ERROR: SCRIPT_NAME environment variable is required"\n\
    exit 1\n\
fi\n\
echo "Starting DeFi Pipeline - Script: $SCRIPT_NAME"\n\
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"\n\
echo "PYTHONPATH: $PYTHONPATH"\n\
exec python pipeline_runner.py "$SCRIPT_NAME"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && chown appuser:appuser /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["/app/entrypoint.sh"]