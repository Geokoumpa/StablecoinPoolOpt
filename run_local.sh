#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Local DeFi Pipeline...${NC}"

# Check if docker compose is running
if ! docker compose ps | grep -q "runner-lightweight"; then
    echo -e "${BLUE}Docker stack not running. Starting up...${NC}"
    docker compose up -d
    echo -e "${GREEN}Docker stack started.${NC}"
    echo "Waiting for services to be ready..."
    sleep 5
fi

# Install local dependencies if needed
if ! python3 -c "import yaml" &> /dev/null; then
    echo "Installing PyYAML for local orchestrator..."
    pip install pyyaml
fi

# Run the orchestrator
echo -e "${BLUE}📝 Parsing workflow.yaml and executing steps...${NC}"
python3 local_orchestrator.py

echo -e "${GREEN}✅ Local run completed.${NC}"
