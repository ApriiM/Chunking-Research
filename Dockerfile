FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System tools used by repo scripts and optional dataset/model utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in layers for better cache reuse.
COPY requirements.txt /app/requirements.txt

# Keep explicit CUDA-enabled PyTorch install as requested.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir --upgrade sentence-transformers transformers accelerate datasets optimum && \
    pip install --no-cache-dir -r /app/requirements.txt

# Runtime NLP resources needed by some chunkers.
RUN python -m nltk.downloader punkt punkt_tab && \
    python -m spacy download en_core_web_sm

# Copy repository (scripts/src/configs/data/processed, etc.) into image.
COPY . /app

CMD ["/bin/bash"]
