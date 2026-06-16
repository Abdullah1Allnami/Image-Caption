#!/bin/bash

# Define python environment paths
PYTHON_ENV="/opt/anaconda3/envs/vit_env/bin/python"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting HTML/JS Frontend with Flask on Port 5004..."
"$PYTHON_ENV" "$DIR/server.py"
