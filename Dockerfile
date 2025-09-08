FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ -z "$SCRIPT_NAME" ]; then\n\
    echo "SCRIPT_NAME environment variable is required"\n\
    exit 1\n\
fi\n\
echo "Running script: $SCRIPT_NAME"\n\
python main_pipeline.py' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]