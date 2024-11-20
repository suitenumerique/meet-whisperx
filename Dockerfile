FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Upgrade system packages to install security updates
RUN apt-get update && \
    apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*


FROM base AS builder

# Install Python and Pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /builder
COPY . /builder


# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir .

# wip try installing flash attn
RUN python3 -m pip install packaging ninja
RUN python3 -m pip install flash-attn --no-build-isolation


FROM base AS core

COPY --from=builder /usr/local /usr/local

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    python3 && \
    rm -rf /var/lib/apt/lists/*

ARG DOCKER_USER=1000
RUN groupadd -r whisperuser && \
    useradd -r -g whisperuser -u ${DOCKER_USER} whisperuser

# wip not sure about this part
ARG HF_HOME=/data/models
ENV HF_HOME=${HF_HOME}

# wip not sure about this part
RUN mkdir -p ${HF_HOME} && \
    chown -R whisperuser:whisperuser ${HF_HOME} && \
    chmod -R 766 ${HF_HOME}

USER whisperuser

WORKDIR /app
COPY --chown=appuser:appuser ./app /app
COPY  ./logging-config.yaml /app/logging-config.yaml

CMD ["python3", "main.py", "--model", "openai/whisper-large-v3", "--logging-config", "logging-config.yaml"]
