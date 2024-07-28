#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Function to download and verify file
download_and_verify() {
    local FILE_PATH=$1
    local URL=$2

    # Create directory if it doesn't exist
    DIR_PATH=$(dirname "$FILE_PATH")
    mkdir -p "$DIR_PATH"

    # Remove existing file if it exists
    if [ -f "$FILE_PATH" ]; then
        rm "$FILE_PATH"
    fi

    # Download file
    wget -q -O "$FILE_PATH" "$URL" || { echo "Failed to download $FILE_PATH"; exit 1; }

    # Verify download
    if [ ! -f "$FILE_PATH" ]; then
        echo "Failed to download $FILE_PATH"
        exit 1
    fi
}

# Download necessary files
download_and_verify /comfyui/styles.csv https://huggingface.co/Gerchegg/TestModel/resolve/main/styles.csv
download_and_verify /comfyui/extra_model_paths.yaml https://huggingface.co/Gerchegg/TestModel/resolve/main/extra_model_paths.yaml

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "runpod-worker-comfy: Starting ComfyUI"
    python3 /comfyui/main.py --disable-auto-launch --disable-metadata --listen &

    echo "runpod-worker-comfy: Starting RunPod Handler"
    python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "runpod-worker-comfy: Starting ComfyUI"
    python3 /comfyui/main.py --disable-auto-launch --disable-metadata &

    echo "runpod-worker-comfy: Starting RunPod Handler"
    python3 -u /rp_handler.py
fi
