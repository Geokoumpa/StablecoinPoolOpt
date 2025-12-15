#!/bin/bash

echo "Scanning the entire project tree with vulture..."

# Run vulture on the current directory recursively
# Exclude common virtual environment names and artifacts
vulture . --exclude .venv,venv,env,.git,__pycache__,.pytest_cache,build,dist,web-ui/node_modules > vulture_report.txt

echo "Scan complete. Report saved to vulture_report.txt"