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
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone https://github.com/Gourieff/comfyui-reactor-node custom_nodes/comfyui-reactor-node || (cd custom_nodes/comfyui-reactor-node && git pull)
RUN git clone https://github.com/chrisgoringe/cg-use-everywhere custom_nodes/cg-use-everywhere
RUN git clone https://github.com/shadowcz007/comfyui-mixlab-nodes custom_nodes/comfyui-mixlab-nodes
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use custom_nodes/ComfyUI-Easy-Use
RUN git clone https://github.com/jitcoder/lora-info custom_nodes/lora-info
RUN git clone https://github.com/StartHua/Comfyui_joytag custom_nodes/Comfyui_joytag
RUN git clone https://github.com/shiimizu/ComfyUI_smZNodes custom_nodes/ComfyUI_smZNodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/theUpsider/ComfyUI-Styles_CSV_Loader custom_nodes/ComfyUI-Styles_CSV_Loader
RUN git clone https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus.git custom_nodes/ComfyUI-PhotoMaker-Plus
RUN git clone https://github.com/shockz0rz/comfy-easy-grids custom_nodes/comfy-easy-grids
RUN git clone https://github.com/sipherxyz/comfyui-art-venture custom_nodes/comfyui-art-venture
RUN git clone https://github.com/PCMonsterx/ComfyUI-CSV-Loader custom_nodes/ComfyUI-CSV-Loader
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/lama/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/clip_interrogator/Salesforce/blip-image-captioning-base/pytorch_model.bin https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/prompt_generator/text2image-prompt-generator/model.safetensors https://huggingface.co/succinctly/text2image-prompt-generator/resolve/main/model.safetensors
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/prompt_generator/opus-mt-zh-en/pytorch_model.bin https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/resolve/main/pytorch_model.bin
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/custom_nodes/Comfyui_joytag/checkpoints/model.safetensors https://huggingface.co/fancyfeast/joytag/resolve/main/model.safetensors
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/custom_nodes/Comfyui_joytag/checkpoints/model.onnx https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/custom_nodes/Comfyui_joytag/checkpoints/model.pt https://huggingface.co/fancyfeast/joytag/resolve/main/model.pt
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/styles.csv https://huggingface.co/Gerchegg/TestModel/resolve/main/styles.csv
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/facedetection/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/photomaker/photomaker-v1.bin https://huggingface.co/TencentARC/PhotoMaker/resolve/main/photomaker-v1.bin
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/sams/sam_vit_b_01ec64.pth https://huggingface.co/segments-arnaud/sam_vit_b/resolve/f38484d6934e5d2b555b1685d22d676236455685/sam_vit_b_01ec64.pth
RUN wget -q --show-progress --progress=bar:force:noscroll -O /workspace/ComfyUI/models/mediapipe/selfie_multiclass_256x256.tflite https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite
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

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
