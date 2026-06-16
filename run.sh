#!/bin/bash

# Define python environment paths
PYTHON_ENV="/opt/anaconda3/envs/vit_env/bin/python"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================="
echo "  Show & Tell Image Captioning Service Runner    "
echo "================================================="
echo "  1) Start HTML/JS Frontend (Flask - Port 5003)"
echo "  2) Download/Prepare Dataset"
echo "================================================="
read -p "Select option (1-2): " option

case $option in
    1)
        echo "Starting HTML/JS Frontend with Flask..."
        "$PYTHON_ENV" "$DIR/server.py"
        ;;
    2)
        read -p "Prepare full Flickr8k dataset (1GB) or demo (random images)? (full/demo) [demo]: " mode
        if [[ "$mode" == "full" ]]; then
            echo "Downloading and extracting full Flickr8k dataset..."
            "$PYTHON_ENV" "$DIR/download_data.py"
        else
            echo "Preparing demo dataset..."
            "$PYTHON_ENV" "$DIR/download_data.py" --demo
        fi
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
