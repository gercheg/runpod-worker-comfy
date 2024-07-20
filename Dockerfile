# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

ARG SKIP_DEFAULT_MODELS
# Download checkpoints/vae/LoRA to include in image.
RUN if [ -z "$SKIP_DEFAULT_MODELS" ]; then wget -O models/checkpoints/sexoholicRealPony_sexoholicRealPony.safetensors "https://huggingface.co/Gerchegg/TestModel/resolve/main/checkpoints/sexoholicRealPony_sexoholicRealPony.safetensors"; fi
RUN if [ -z "$SKIP_DEFAULT_MODELS" ]; then wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors; fi
# Download and setup custom nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone https://github.com/Gourieff/comfyui-reactor-node custom_nodes/comfyui-reactor-node
RUN git clone https://github.com/chrisgoringe/cg-use-everywhere custom_nodes/cg-use-everywhere
RUN git clone https://github.com/shadowcz007/comfyui-mixlab-nodes custom_nodes/comfyui-mixlab-nodes
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use custom_nodes/ComfyUI-Easy-Use
RUN git clone https://github.com/jitcoder/lora-info custom_nodes/lora-info
RUN git clone https://github.com/StartHua/Comfyui_joytag custom_nodes/Comfyui_joytag
RUN git clone https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node
RUN git clone https://github.com/shiimizu/ComfyUI_smZNodes custom_nodes/ComfyUI_smZNodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack

# Install ComfyUI dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt \
    && pip3 install ultralytics==8.2.55 \
    && pip3 install numexpr==2.10.1 

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./
# Support styles.csv
ADD src/styles.csv ./
# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
