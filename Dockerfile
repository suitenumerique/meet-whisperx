FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Upgrade system packages to install security updates
RUN apt-get update -y && \
  apt-get -y upgrade && \
  apt-get install -y ffmpeg

RUN apt install python3 -y
RUN apt install python3-pip -y

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app