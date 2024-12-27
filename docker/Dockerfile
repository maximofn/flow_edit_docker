FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs python3 python3-pip \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && git lfs install \
    && rm cuda-keyring_1.0-1_all.deb \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone --depth 1 https://huggingface.co/spaces/fallenshock/FlowEdit \
    && cd FlowEdit && pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir gradio spaces python-dotenv

RUN sed -i 's/from diffusers import FluxPipeline, StableDiffusion3Pipeline/from diffusers import DiffusionPipeline/g' /FlowEdit/app.py \
    && sed -i 's|pipe_sd3 = |import dotenv # pipe_sd3 = |g' /FlowEdit/app.py \
    && sed -i 's|pipe_flux = |dotenv.load_dotenv() # pipe_flux = |g' /FlowEdit/app.py \
    && sed -i 's/# pipe_sd3.to(device)/HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")/g' /FlowEdit/app.py \
    && sed -i 's/# pipe_flux.to(device)/pipe = DiffusionPipeline.from_pretrained("black-forest-labs\/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced", token=HUGGING_FACE_TOKEN)/g' /FlowEdit/app.py \
    && sed -i 's/pipe = pipe_flux.to(device)/pass/g' /FlowEdit/app.py \
    && sed -i 's/pipe = pipe_sd3.to(device)/pass/g' /FlowEdit/app.py \
    && sed -i 's/, value="SD3")/, value="FLUX")/g' /FlowEdit/app.py

WORKDIR /FlowEdit
