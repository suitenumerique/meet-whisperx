FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update -y && \
  apt-get -y upgrade && \
  apt-get install -y ffmpeg
RUN apt install python3 -y
RUN apt install python3-pip -y

USER whisper
RUN groupadd --gid 1100 whisper
RUN useradd --home /home/whisper --gid 1100 --uid 1100 whisper

WORKDIR /home/whisper
RUN ls
RUN sleep 10
ADD ./pyproject.toml ./pyproject.toml
RUN pip install ".[app]"
ADD ./app /home/whisper/app
ENV PYTHONPATH="/home/whisper/app:${PYTHONPATH}"

# Set home to the user's home directory
#ENV HOME=/home/user
#ENV PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
#WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
#COPY --chown=user . $HOME/app