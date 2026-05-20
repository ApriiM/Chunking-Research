# CUDA 12.6 + cuDNN — matches torch==2.7.1+cu126
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps for sentence-transformers / tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only metadata first (lepszy cache warstw)
COPY pyproject.toml ./

# Instalacja zależności (BEZ optionala need_torch)
RUN pip install --no-cache-dir .

# Dopiero teraz kod
COPY app.py .

# Cache modeli
ENV TRANSFORMERS_CACHE=/model-cache
ENV HF_HOME=/model-cache
ENV SENTENCE_TRANSFORMERS_HOME=/model-cache

ENV SIM_PORT=5051

EXPOSE 5051

CMD ["python", "app.py"]