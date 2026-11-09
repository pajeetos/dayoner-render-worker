# Dayoner render worker for RunPod Serverless: Blender 5.2 on the GPU (Eevee, or
# Cycles where Eevee cannot open EGL), ffmpeg, Python with Whisper, and Node for
# the upload. No render scripts live here: every job brings its own (handler.py).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Eevee needs the driver's EGL/OpenGL libraries, not only CUDA: ask the
    # container runtime for every driver capability, graphics included.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ffmpeg fonts-dejavu-core curl ca-certificates xz-utils \
      libegl1 libgl1 libgles2 libglvnd0 libglx0 libopengl0 libvulkan1 \
      libx11-6 libxext6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxrandr2 libxinerama1 libxcursor1 \
      libxkbcommon0 libsm6 libice6 libwayland-client0 libwayland-egl1 libdbus-1-3 libfontconfig1 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# glvnd finds the NVIDIA EGL the host mounts through this vendor file.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ARG BLENDER_VERSION=5.2.3
RUN curl -fsSL "https://download.blender.org/release/Blender5.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz" | tar -xJ -C /opt \
 && mv "/opt/blender-${BLENDER_VERSION}-linux-x64" /opt/blender
ENV BLENDER_BIN=/opt/blender/blender

# Node 20+ for @vercel/blob (Ubuntu ships 18).
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Word timings for lip sync: the model is baked in so a cold worker does not download it.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY package.json .
RUN npm install --omit=dev --no-audit --no-fund
COPY handler.py upload.mjs ./

CMD ["python", "-u", "handler.py"]
