FROM python:3.12-slim

RUN groupadd --gid 1100 whisper
RUN useradd --home /home/whisper --gid 1100 --uid 1100 whisper 
USER whisper
ENV PATH="/home/whisper/.local/bin:${PATH}"

WORKDIR /home/whisper
ADD ./pyproject.toml ./pyproject.toml
RUN pip install .
ADD ./app /home/whisper/app
ENV PYTHONPATH="/home/whisper/app:${PYTHONPATH}"
