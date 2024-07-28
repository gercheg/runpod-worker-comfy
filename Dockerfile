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
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

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
RUN mkdir -p /comfyui/models/lama && \
    ( [ ! -f /comfyui/models/lama/big-lama.pt ] || rm /comfyui/models/lama/big-lama.pt ) && \
    wget -q -O /comfyui/models/lama/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt && \
    test -f /comfyui/models/lama/big-lama.pt || { echo "Failed to download big-lama.pt"; exit 1; }

RUN mkdir -p /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base && \
    ( [ ! -f /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin ] || rm /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin ) && \
    wget -q -O /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin && \
    test -f /comfyui/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin || { echo "Failed to download pytorch_model.bin"; exit 1; }

RUN mkdir -p /comfyui/models/prompt_generator/opus-mt-zh-en && \
    ( [ ! -f /comfyui/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin ] || rm /comfyui/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin ) && \
    wget -q -O /comfyui/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin https://huggingface.co/Gerchegg/TestModel/resolve/main/opus-mt-zh-en/pytorch_model.bin && \
    test -f /comfyui/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin || { echo "Failed to download opus-mt-zh-en/pytorch_model.bin"; exit 1; }

RUN mkdir -p /comfyui/custom_nodes/Comfyui_joytag/checkpoints && \
    ( [ ! -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors ] || rm /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors ) && \
    wget -q -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors https://huggingface.co/Gerchegg/TestModel/resolve/main/Comfyui_joytag/checkpoints/model.safetensors && \
    test -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors || { echo "Failed to download model.safetensors"; exit 1; }

RUN mkdir -p /comfyui/custom_nodes/Comfyui_joytag/checkpoints && \
    ( [ ! -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.onnx ] || rm /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.onnx ) && \
    wget -q -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.onnx https://huggingface.co/Gerchegg/TestModel/resolve/main/Comfyui_joytag/checkpoints/model.onnx && \
    test -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.onnx || { echo "Failed to download model.onnx"; exit 1; }

RUN mkdir -p /comfyui/custom_nodes/Comfyui_joytag/checkpoints && \
    ( [ ! -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.pt ] || rm /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.pt ) && \
    wget -q -O /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.pt https://huggingface.co/Gerchegg/TestModel/resolve/main/Comfyui_joytag/checkpoints/model.pt && \
    test -f /comfyui/custom_nodes/Comfyui_joytag/checkpoints/model.pt || { echo "Failed to download model.pt"; exit 1; }

RUN mkdir -p /comfyui/models/ultralytics/bbox && \
    ( [ ! -f /comfyui/models/ultralytics/bbox/face_yolov8m.pt ] || rm /comfyui/models/ultralytics/bbox/face_yolov8m.pt ) && \
    wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt && \
    test -f /comfyui/models/ultralytics/bbox/face_yolov8m.pt || { echo "Failed to download face_yolov8m.pt"; exit 1; }

RUN mkdir -p /comfyui/models/facedetection && \
    ( [ ! -f /comfyui/models/facedetection/detection_Resnet50_Final.pth ] || rm /comfyui/models/facedetection/detection_Resnet50_Final.pth ) && \
    wget -q -O /comfyui/models/facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    test -f /comfyui/models/facedetection/detection_Resnet50_Final.pth || { echo "Failed to download detection_Resnet50_Final.pth"; exit 1; }

RUN mkdir -p /comfyui/models/facedetection && \
    ( [ ! -f /comfyui/models/facedetection/parsing_parsenet.pth ] || rm /comfyui/models/facedetection/parsing_parsenet.pth ) && \
    wget -q -O /comfyui/models/facedetection/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth && \
    test -f /comfyui/models/facedetection/parsing_parsenet.pth || { echo "Failed to download parsing_parsenet.pth"; exit 1; }

RUN mkdir -p /comfyui/models/photomaker && \
    ( [ ! -f /comfyui/models/photomaker/photomaker-v1.bin ] || rm /comfyui/models/photomaker/photomaker-v1.bin ) && \
    wget -q -O /comfyui/models/photomaker/photomaker-v1.bin https://huggingface.co/TencentARC/PhotoMaker/resolve/main/photomaker-v1.bin && \
    test -f /comfyui/models/photomaker/photomaker-v1.bin || { echo "Failed to download photomaker-v1.bin"; exit 1; }

RUN mkdir -p /comfyui/models/sams && \
    ( [ ! -f /comfyui/models/sams/sam_vit_b_01ec64.pth ] || rm /comfyui/models/sams/sam_vit_b_01ec64.pth ) && \
    wget -q -O /comfyui/models/sams/sam_vit_b_01ec64.pth https://huggingface.co/segments-arnaud/sam_vit_b/resolve/f38484d6934e5d2b555b1685d22d676236455685/sam_vit_b_01ec64.pth && \
    test -f /comfyui/models/sams/sam_vit_b_01ec64.pth || { echo "Failed to download sam_vit_b_01ec64.pth"; exit 1; }

RUN mkdir -p /comfyui/models/mediapipe && \
    ( [ ! -f /comfyui/models/mediapipe/selfie_multiclass_256x256.tflite ] || rm /comfyui/models/mediapipe/selfie_multiclass_256x256.tflite ) && \
    wget -q -O /comfyui/models/mediapipe/selfie_multiclass_256x256.tflite https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite && \
    test -f /comfyui/models/mediapipe/selfie_multiclass_256x256.tflite || { echo "Failed to download selfie_multiclass_256x256.tflite"; exit 1; }
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
