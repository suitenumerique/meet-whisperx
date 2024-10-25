FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update -y && \
  apt-get -y upgrade && \
  apt-get install -y ffmpeg python3 python3-pip python3-dev curl && \
  rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1100 whisper
RUN useradd --home /home/whisper --gid 1100 --uid 1100 whisper 
USER whisper
ENV PATH="/home/whisper/.local/bin:${PATH}"

WORKDIR /home/whisper
ADD ./pyproject.toml ./pyproject.toml
RUN pip install .
ADD ./app /home/whisper/app
ENV PYTHONPATH="/home/whisper/app:${PYTHONPATH}"
