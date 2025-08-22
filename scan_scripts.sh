#!/bin/bash

# Find all unique script paths from main_pipeline.py
scripts=$(grep -oP "run_script\(\"\K[^\"]+" main_pipeline.py | sed 's/\./\//g' | sort -u)

# Run vulture on each script and append the output to a file
for script in $scripts; do
    echo "Scanning $script.py..."
    vulture "$script.py" >> vulture_report.txt
done

echo "Scan complete. Report saved to vulture_report.txt"