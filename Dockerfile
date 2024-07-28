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
# Download and setup custom nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager \
    && git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui \
    && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && git clone https://github.com/Gourieff/comfyui-reactor-node custom_nodes/comfyui-reactor-node || (cd custom_nodes/comfyui-reactor-node && git pull) \
    && git clone https://github.com/chrisgoringe/cg-use-everywhere custom_nodes/cg-use-everywhere \
    && git clone https://github.com/shadowcz007/comfyui-mixlab-nodes custom_nodes/comfyui-mixlab-nodes \
    && git clone https://github.com/yolain/ComfyUI-Easy-Use custom_nodes/ComfyUI-Easy-Use \
    && git clone https://github.com/jitcoder/lora-info custom_nodes/lora-info \
    && git clone https://github.com/StartHua/Comfyui_joytag custom_nodes/Comfyui_joytag \
    && git clone https://github.com/shiimizu/ComfyUI_smZNodes custom_nodes/ComfyUI_smZNodes \
    && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack \
    && git clone https://github.com/theUpsider/ComfyUI-Styles_CSV_Loader custom_nodes/ComfyUI-Styles_CSV_Loader \
    && git clone https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus.git custom_nodes/ComfyUI-PhotoMaker-Plus \
    && git clone https://github.com/shockz0rz/comfy-easy-grids custom_nodes/comfy-easy-grids \
    && git clone https://github.com/sipherxyz/comfyui-art-venture custom_nodes/comfyui-art-venture \
    && git clone https://github.com/PCMonsterx/ComfyUI-CSV-Loader custom_nodes/ComfyUI-CSV-Loader


# Go back to the root
WORKDIR /

# Download necessary models and files
RUN wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/lama/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/prompt_generator/text2image-prompt-generator/model.safetensors https://huggingface.co/succinctly/text2image-prompt-generator/resolve/main/model.safetensors \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/pytorch_model.bin \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors https://huggingface.co/fancyfeast/joytag/resolve/main/model.safetensors \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.onnx https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.pt https://huggingface.co/fancyfeast/joytag/resolve/main/model.pt \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/facedetection/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/photomaker/photomaker-v1.bin https://huggingface.co/TencentARC/PhotoMaker/resolve/main/photomaker-v1.bin \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/sams/sam_vit_b_01ec64.pth https://huggingface.co/segments-arnaud/sam_vit_b/resolve/f38484d6934e5d2b555b1685d22d676236455685/sam_vit_b_01ec64.pth \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/models/mediapipe/selfie_multiclass_256x256.tflite https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/styles.csv https://huggingface.co/Gerchegg/TestModel/resolve/main/styles.csv \
    && wget -q --show-progress --progress=bar:force:noscroll -O /comfyui/extra_model_paths.yaml https://huggingface.co/Gerchegg/TestModel/resolve/main/extra_model_paths.yaml


# Install ComfyUI dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt \
    && pip3 install ultralytics==8.2.55 \
    && pip3 install numexpr==2.10.1 

# Install runpod
RUN pip3 install runpod requests

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
